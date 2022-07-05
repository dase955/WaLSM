// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

#include <time.h>
#include <sys/time.h>

#include <unordered_set>
#include <algorithm>
#include <mutex>
#include <random>
#include <thread>
#include <cmath>
#include <time.h>

using namespace rocksdb;

enum Operation {
  kRead,
  kWrite,
};

class BaseKeyGenerator {
 public:
  BaseKeyGenerator(int record_count, uint64_t sample_range)
      : interval_length_(sample_range / 100),
        record_count_(record_count),
        operation_count_(record_count),
        sample_range_(sample_range) {
    InitializeIntervals();
  };

  virtual ~BaseKeyGenerator() {}

  virtual void Prepare() = 0;

  virtual std::vector<uint64_t>& GetRecords() = 0;

  void SetOperationCount(int operation_count) {
    if (operation_count > record_count_) {
      printf("Error: operation count must less than record count\n");
      abort();
    }
    operation_count_ = operation_count;
  }

  int GetOperationCount() const {
    return operation_count_;
  }

  virtual Operation Next(std::string& key) = 0;

  virtual std::string NextReadKey() = 0;

  virtual std::string NextWriteKey() = 0;

  virtual void AddWrittenKey(uint64_t v) = 0;

  void SetReadRatio(double read_ratio) {
    read_ratio_ = read_ratio;
  }

  void SetPrefix(std::string& prefix) {
    prefix_ = prefix;
  }

 protected:
  void InitializeIntervals() {
    for (int i = 0; i < 100; ++i) {
      intervals_[i] = interval_length_ * i;
    }

    std::random_device shuffle_rd;
    std::vector<int> r(100);
    std::iota(r.begin(), r.end(), 0);
    std::shuffle(r.begin(), r.end(), shuffle_rd);
    for (int i = 0; i < 10; ++i) {
      hot_read_intervals_.push_back(r[i]);
      hot_write_intervals_.push_back(r[i]);
    }
    for (int i = 10; i < 20; ++i) {
      hot_read_intervals_.push_back(r[i]);
    }
    for (int i = 20; i < 30; ++i) {
      hot_write_intervals_.push_back(r[i]);
    }
  }

  double read_ratio_ = 0.5;

  uint64_t intervals_[100];

  std::vector<int> hot_read_intervals_;

  std::vector<int> hot_write_intervals_;

  std::vector<uint64_t> hot_keys_written_;

  std::vector<uint64_t> cold_keys_written_;

  uint64_t interval_length_;

  int record_count_;

  int operation_count_;

  uint64_t sample_range_;

  std::string prefix_ = "user";
};

class YCSBLoadGenerator : public BaseKeyGenerator {
 public:
  YCSBLoadGenerator(int record_count, uint64_t sample_range)
      : BaseKeyGenerator(record_count, sample_range) {};

  void Prepare() override {
    if (records_.empty()) {
      records_.resize(record_count_);
      for (int i = 0; i < record_count_; ++i) {
        records_[i] = fnvhash64(i);
      }
    }
    cur_index_.store(0, std::memory_order_relaxed);
  }

  Operation Next(std::string& key) override {
    key = NextWriteKey();
    return kWrite;
  }

  std::string NextReadKey() override {
    printf("Why you do read operation in YCSBLoadGenerator?\n");
    abort();
    return "";
  }

  std::string NextWriteKey() override {
    auto s = std::to_string(records_[cur_index_++]);
    return prefix_ + std::string(19 - s.length(), '0') + s;
  }

  void AddWrittenKey(uint64_t v) override {
    printf("There is not need to add key in YCSBLoadGenerator?\n");
  }

  std::vector<uint64_t>& GetRecords() override {
    return records_;
  }

 private:
  static uint64_t fnvhash64(int64_t val) {
    //from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
    static int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325LL;
    static int64_t FNV_PRIME_64 = 1099511628211L;

    int64_t hashval = FNV_OFFSET_BASIS_64;

    for (int i = 0; i < 8; i++) {
      int64_t octet = val & 0x00ff;
      val = val >> 8;

      hashval = hashval ^ octet;
      hashval = hashval * FNV_PRIME_64;
      //hashval = hashval ^ octet;
    }
    return hashval > 0 ? hashval : -hashval;
  }

  std::vector<uint64_t> records_;

  std::atomic<int> cur_index_{0};
};

class ZipfianGenerator : public BaseKeyGenerator {
 public:
  explicit ZipfianGenerator(int record_count,
                            uint64_t sample_range = 9000000000000000000UL,
                            double alpha = 0.98)
      : BaseKeyGenerator(record_count, sample_range), alpha_(alpha) {
    std::random_device rd;
    std::random_device rd2;
    rng_ = std::default_random_engine{rd()};
    rng2_ = std::default_random_engine{rd2()};

    write_records_.resize(record_count);
    hot_keys_written_.reserve(record_count * 2);
    cold_keys_written_.reserve(record_count * 2);
  }

  void Prepare() override {
    write_index_.store(0, std::memory_order_relaxed);
    double c = 0.0;

    uint64_t zipf_sample_range = 1000000000;

    for (size_t i = 1; i <= zipf_sample_range; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    double* sum_probs = new double[zipf_sample_range + 1];
    sum_probs[0] = 0;
    for (size_t i = 1; i <= zipf_sample_range; ++i) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha_);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_real_distribution<> rand_val{0.0, 1.0};

    printf("Generate samples\n");

    std::unordered_map<uint64_t, int> freqs;
    std::vector<uint64_t> records;
    records.resize(record_count_);
    for (int i = 0; i < record_count_; ++i) {
      uint64_t res = 0;
      double z;
      do {
        z = rand_val(rng);
      } while ((z == 0) || (z == 1));

      uint64_t low = 1;
      uint64_t high = zipf_sample_range;
      uint64_t mid;
      do {
        mid = (low + high) / 2;
        if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
          res = mid;
          break;
        } else if (sum_probs[mid] >= z) {
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      } while (low <= high);

      ++freqs[res];
      records[i] = res;
    }

    CheckFreq(records);

    GenerateWriteSamples(records, freqs);
    delete[] sum_probs;

    CheckFreq(write_records_);
  }

  Operation Next(std::string& key) override {
    if (rand_double(rng_) < read_ratio_) {
      key = NextReadKey();
      return kRead;
    } else {
      key = NextWriteKey();
      return kWrite;
    }
  }

  std::string NextReadKey() override {
    static std::uniform_int_distribution<uint64_t> rand1(
        0, hot_keys_written_.size() - 1);
    static std::uniform_int_distribution<uint64_t> rand2(
        0, cold_keys_written_.size() - 1);

    uint64_t v;
    if (rand_double(rng_) < 0.2) {
      v = hot_keys_written_[rand1(rng2_)];
    } else {
      v = cold_keys_written_[rand2(rng2_)];
    }

    auto s = std::to_string(v);
    return prefix_ + std::string(19 - s.length(), '0') + s;
  }

  std::string NextWriteKey() override {
    auto s = std::to_string(write_records_[write_index_++]);
    return prefix_ + std::string(19 - s.length(), '0') + s;
  }

  void AddWrittenKey(uint64_t v) override {
    int i = std::min(v / interval_length_, 99ul);
    if (hot_read_intervals_[i]) {
      hot_keys_written_.push_back(v);
    } else {
      cold_keys_written_.push_back(v);
    }
  }

  std::vector<uint64_t>& GetRecords() override {
    return write_records_;
  }

 private:
  static uint64_t fnvhash64(int64_t val) {
    //from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
    static int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325LL;
    static int64_t FNV_PRIME_64 = 1099511628211L;

    int64_t hashval = FNV_OFFSET_BASIS_64;

    for (int i = 0; i < 8; i++) {
      int64_t octet = val & 0x00ff;
      val = val >> 8;

      hashval = hashval ^ octet;
      hashval = hashval * FNV_PRIME_64;
      //hashval = hashval ^ octet;
    }
    return hashval > 0 ? hashval : -hashval;
  }

  void GenerateWriteSamples(
      std::vector<uint64_t>& records,
      std::unordered_map<uint64_t, int>& freqs) {
    std::unordered_map<uint64_t, uint64_t> modified;

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_int_distribution<uint64_t> dist(0, sample_range_ - 1);

    auto check_func = [&](uint64_t val) {
      for (auto i : hot_write_intervals_) {
        if (val >= intervals_[i] && val < intervals_[i] + interval_length_) {
          return false;
        }
      }
      return true;
    };

    std::unordered_set<uint64_t> sampled;
    uint64_t mod = interval_length_ * 20;

    int cur = 0;
    for (auto& pair : freqs) {
      uint64_t old_value = pair.first;
      uint64_t new_value;

      if (pair.second == 1) {
        new_value = dist(rng);
        while (!check_func(new_value)) {
          new_value = dist(rng);
        }
      } else {
        new_value = fnvhash64(old_value) % mod;
        while (sampled.find(new_value) != sampled.end()) {
          new_value = fnvhash64(new_value) % mod;
        }

        int i = hot_write_intervals_[new_value / interval_length_];
        new_value = intervals_[i] + (new_value % interval_length_);
        sampled.emplace(new_value);
      }

      modified[old_value] = new_value;
      AddWrittenKey(new_value);
    }

    for (int i = 0; i < record_count_; ++i) {
      write_records_[i] = modified[records[i]];
    }
  }

  static void CheckFreq(std::vector<uint64_t>& samples) {
    std::cout << "Check Freq start" << std::endl;

    std::unordered_map<uint64_t, int> freqs;
    for (auto& value : samples) {
      ++freqs[value];
    }

    int cold_count = 0;
    for (auto& pair : freqs) {
      cold_count += (pair.second == 1);
    }

    int hot_count = freqs.size() - cold_count;
    float hot_freq = (samples.size() - cold_count) / (float)samples.size() * 100.f;
    printf("Sample generated. hot count = %d(%.2f), cold count = %d\n",
           hot_count, hot_freq, cold_count);
    std::cout << "Check Freq end" << std::endl;
  }

  double alpha_;

  std::vector<uint64_t> write_records_;

  std::atomic<int> write_index_{0};

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng_;

  std::default_random_engine rng2_;
};

class WorkloadRunner {
 public:
  WorkloadRunner(int num_threads, DB* db)
      : num_threads_(num_threads), db_(db) {
    work_threads_ = new std::thread[num_threads];
  }

  void SetLoadGenerator(BaseKeyGenerator* generator) {
    load_key_generator_ = generator;
  }

  void SetRunGenerator(BaseKeyGenerator* generator) {
    run_key_generator_ = generator;
  }

  void Load() {
    load_key_generator_->Prepare();
    completed_count_.store(0, std::memory_order_relaxed);

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i] = std::thread(
          &WorkloadRunner::Insert, this, i, load_key_generator_);
    }

    printf("Load opearation count = %d\n", load_key_generator_->GetOperationCount());

    auto ops_thread = std::thread(
        &WorkloadRunner::StatisticsThread, this,
        load_key_generator_->GetOperationCount());
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i].join();
    }
  }

  void Run() {
    run_key_generator_->Prepare();

    for (auto& v : load_key_generator_->GetRecords()) {
      run_key_generator_->AddWrittenKey(v);
    }

    completed_count_.store(0, std::memory_order_relaxed);

    printf("Run opearation count = %d\n", run_key_generator_->GetOperationCount());

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i] = std::thread(
          &WorkloadRunner::Insert, this, i, run_key_generator_);
    }

    auto ops_thread = std::thread(
        &WorkloadRunner::StatisticsThread, this,
        run_key_generator_->GetOperationCount());
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      work_threads_[i].join();
    }
  }

  void SetMetricInterval(int interval) {
    interval_ = interval;
  }

  static std::string GenerateValueFromKey(std::string& key) {
    int repeat_times = 1024UL / key.length();
    size_t len = key.length() * repeat_times;
    std::string value;
    value.reserve(1024);
    while (repeat_times--) {
      value += key;
    }
    value += key.substr(0, 1024 - value.length());
    return value;
  }

 private:
  // TODO: need latency statistics
  void Insert(int thread_id, BaseKeyGenerator* generator) {
    int operation_count = generator->GetOperationCount();
    int operation_per_thread = operation_count / num_threads_;
    if (thread_id == 0) {
      operation_per_thread += (operation_count % num_threads_);
    }

    std::string key;
    std::string ret;
    Status s;

    while (operation_per_thread--) {
      auto type = generator->Next(key);
      auto value = GenerateValueFromKey(key);

      switch (type) {
        case kWrite:
          s = db_->Put(WriteOptions(), key, value);
          break;
        case kRead:
          s = db_->Get(ReadOptions(), key, &ret);
          break;
      }

      ++completed_count_;
      assert(s.ok() || s.IsNotFound());
    }
  }

  void StatisticsThread(int operation_counts) {
    int prev_completed = 0;
    int seconds = 0;
    while (prev_completed < operation_counts) {
      int new_completed = completed_count_.load(std::memory_order_relaxed);
      int ops = (new_completed - prev_completed) / interval_;
      printf("[%d sec] %d operations; %d current ops/sec\n",
             seconds, new_completed, ops);

      prev_completed = new_completed;
      std::this_thread::sleep_for(std::chrono::seconds(interval_));
      seconds += interval_;
    }
  }

  int interval_ = 5;

  int num_threads_ = 16;

  std::thread* work_threads_;

  std::thread ops_thread_;

  std::atomic<int> completed_count_{0};

  std::atomic<int> failed_count_{0};

  BaseKeyGenerator* load_key_generator_ = nullptr;

  BaseKeyGenerator* run_key_generator_ = nullptr;

  DB* db_;
};

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  float read_ratio = 0.5f;
  if (argc == 2) {
    read_ratio = atof(argv[1]);
  }

  printf("Read ratio = %.2f\n", read_ratio);

  int thread_num = 8;
  int load_count = 80000000;
  int run_count = 320000000;
  uint64_t sample_range = 9000000000000000000;

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;

  std::string db_path = "/home/crh/db_test_nvm_l0";

  DB* db;
  DB::Open(options, db_path, &db);

  auto load_generator = new YCSBLoadGenerator(load_count, sample_range);
  auto run_generator = new ZipfianGenerator(run_count, sample_range, 0.98);
  run_generator->SetOperationCount(120000000);
  run_generator->SetReadRatio(read_ratio);

  WorkloadRunner runner(thread_num, db);
  runner.SetLoadGenerator(load_generator);
  runner.SetRunGenerator(run_generator);
  runner.Load();
  runner.Run();

  db->Close();

  delete db;
}