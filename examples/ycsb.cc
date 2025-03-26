// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <functional>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "rocksdb/filter_policy.h"

#include <time.h>
#include <sys/time.h>

#include <unordered_set>
#include <algorithm>
#include <mutex>
#include <random>
#include <thread>
#include <cmath>
#include <time.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>


using namespace rocksdb;

std::unordered_map<std::string, int> high_freq_data;

const uint64_t kFNVOffsetBasis64 = 0xCBF29CE484222325ull;
const uint64_t kFNVPrime64 = 1099511628211ull;

inline uint64_t FNVHash64(uint64_t val) {
  uint64_t hash = kFNVOffsetBasis64;

  for (int i = 0; i < 8; i++) {
    uint64_t octet = val & 0x00ff;
    val = val >> 8;

    hash = hash ^ octet;
    hash = hash * kFNVPrime64;
  }
  return hash;
}

class KeyGenerator {
 public:
  KeyGenerator(int record_count, uint64_t sample_range)
      : record_count_(record_count), sample_range_(sample_range) {};

  void SetPrefix(std::string& prefix) {
    prefix_ = prefix;
  }

  virtual uint64_t Next() = 0;

  virtual ~KeyGenerator() {}

  virtual void Prepare() {
      // Do nothing;
  };

  // TODO
  virtual std::string NextKey() {
    std::string s = std::to_string(Next());
    return prefix_ + s;
  }

 protected:
  int record_count_;

  uint64_t sample_range_;

  std::string prefix_ = "user";
};

class YCSBZipfianGenerator : public KeyGenerator{
 public:
  YCSBZipfianGenerator(
      int load_count, int record_count, uint64_t sample_range,
      double theta, double zetan)
      : KeyGenerator(record_count, sample_range),
        theta_(theta), zetan_(zetan), load_count_(load_count) {
    zetan_ = zetastatic(sample_range + 1, theta);
    alpha = 1.0 / (1.0 - theta);
    eta = (1 - std::pow(2.0 / sample_range_, 1 - theta))
          / (1 - zetastatic(2, theta) / zetan_);

    std::mt19937 rd;
    rng = std::default_random_engine{rd()};

    std::cout << "theta = " << theta << ", zetan = " << zetan_ << std::endl;
  }

  static double zetastatic(long n, double theta) {
    double sum = 0.0f;
    long t = n / 100;
    for (long i = 0; i < n; i++) {
      sum += 1 / (std::pow(i + 1, theta));
    }
    return sum;
  }

  void Prepare() override {
    InternalPrepare();
    CheckFreq(samples_);
    PrepareForStats();
  }

  void InternalPrepare() {
    samples_.resize(record_count_);
    std::unordered_map<uint64_t, int> frequency;

    for (int i = 0; i < record_count_; ++i) {
      samples_[i] = Zipf();
      frequency[samples_[i]];
    }

    printf("Distinct count = %zu\n", frequency.size());
    CheckFreq(samples_);

    std::unordered_map<uint64_t, uint64_t> mapper;
    std::vector<std::pair<int, uint64_t>> frequency_sorted;
    frequency_sorted.reserve(frequency.size());
    for (auto& pair : frequency) {
      frequency_sorted.emplace_back(pair.second, pair.first);
    }
    std::sort(frequency_sorted.begin(), frequency_sorted.end(),
              [](const std::pair<int, uint64_t>& p1, const std::pair<int, uint64_t>& p2){
                return p1.first > p2.first;
              });

    size_t idx = 0;
    size_t runs = (frequency.size() + load_count_ - 1) / load_count_;
    for (size_t run = 0; run < runs; ++run) {
      std::mt19937 shuffle_rd;
      std::vector<int> shuffled(load_count_);
      std::iota(std::begin(shuffled), std::end(shuffled), 0);
      std::shuffle(shuffled.begin(), shuffled.end(), shuffle_rd);

      for (size_t i = 0; i < load_count_ && idx < frequency.size(); ++i) {
        mapper[frequency_sorted[idx++].second] = shuffled[i];
      }
    }

    for (auto& sample : samples_) {
      sample = mapper[sample];
    }
    CheckFreq(samples_);
  }

  void PrepareForStats() {
    std::unordered_map<uint64_t, int> freq;
    for (auto& sample : samples_) {
      --freq[sample];
    }
    std::vector<std::pair<int, uint64_t>> pairs;
    for (auto &pair : freq) {
      pairs.emplace_back(pair.second, pair.first);
    }
    std::sort(pairs.begin(), pairs.end());
    int high_count = pairs.size() * 0.1;
    for (int i = 0; i < high_count; ++i) {
      auto v = std::hash<uint64_t>{}(pairs[i].second);
      auto s = "user" + std::to_string(v);
      high_freq_data[s] = -pairs[i].first;
    }
  }

  uint64_t Zipf() {
    double u = rand_double(rng);
    double uz = u * zetan_;
    uint64_t v;
    if (uz < 1.0) {
      v = 0;
    } else if (uz < 1.0 + std::pow(0.5, theta_)) {
      v = 1;
    } else {
      v = (uint64_t) ((sample_range_) * std::pow(eta * u - eta + 1, alpha));
    }
    return fnvhash64(v);
  }

  uint64_t Next() override {
    return FNVHash64(samples_[index_++]);
    //return std::hash<uint64_t>{}(samples_[index_++]);
  }

  static void CheckFreq(std::vector<uint64_t>& samples) {
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
  }

 private:
  static uint64_t fnvhash64(int64_t val) {
    static int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325LL;
    static int64_t FNV_PRIME_64 = 1099511628211L;

    int64_t hashval = FNV_OFFSET_BASIS_64;

    for (int i = 0; i < 8; i++) {
      int64_t octet = val & 0x00ff;
      val = val >> 8;

      hashval = hashval ^ octet;
      hashval = hashval * FNV_PRIME_64;
    }
    return hashval > 0 ? hashval : -hashval;
  }

  double theta_;
  double zetan_;
  double alpha, eta;
  uint64_t load_count_;
  std::atomic<int> index_{0};

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng;

  std::vector<uint64_t> samples_;
};

class YCSBLoadGenerator : public KeyGenerator {
 public:
  YCSBLoadGenerator(int record_count, int sample_range)
      : KeyGenerator(record_count, sample_range) {};

  uint64_t Next() override {
    return FNVHash64(cur_index_++);
  }

 private:
  std::atomic<uint64_t> cur_index_{0};
};

class Inserter {
 public:
  Inserter(int num_threads, DB* db)
      : num_threads_(num_threads), db_(db) {
    insert_threads_ = new std::thread[num_threads];
  }

  void SetGenerator(KeyGenerator* generator) {
    key_generator_ = generator;
  }

  void SetReadProp(float prop) {
    read_prop = prop;
  }

  void SetMetricInterval(int interval) {
    interval_ = interval;
  }

  void SetInsertCount(int count) {
    insert_counts_ = count;
  }

  void DoLoad() {
    key_generator_->Prepare();

    for (int i = 0; i < num_threads_; ++i) {
      insert_threads_[i] = std::thread(&Inserter::Load, this, i);
    }

    auto ops_thread = std::thread(&Inserter::StatisticsThread, this);
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      insert_threads_[i].join();
    }
  }

  void DoInsert() {
    key_generator_->Prepare();

    for (int i = 0; i < num_threads_; ++i) {
      insert_threads_[i] = std::thread(&Inserter::Insert, this, i);
    }

    auto ops_thread = std::thread(&Inserter::StatisticsThread, this);
    ops_thread.join();

    for (int i = 0; i < num_threads_; ++i) {
      insert_threads_[i].join();
    }
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
  void Load(int thread_id) {
    int insert_per_thread = insert_counts_ / num_threads_;
    if (thread_id == 0) {
      insert_per_thread += (insert_counts_ % num_threads_);
    }

    while (insert_per_thread--) {
      [[maybe_unused]] auto begin_time = std::chrono::steady_clock::now();
      auto key = key_generator_->NextKey();
      auto value = GenerateValueFromKey(key);
      Status s = db_->Put(WriteOptions(), key, value);

      ++completed_count_;
      if (!s.ok()) {
        ++failed_count_;
      }
    }
  }

  void Insert(int thread_id) {
    int insert_per_thread = insert_counts_ / num_threads_;
    if (thread_id == 0) {
      insert_per_thread += (insert_counts_ % num_threads_);
    }

    std::mt19937 rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_real_distribution<> rand_val{0.0, 1.0};
    std::string got;
    Status s;

    while (insert_per_thread--) {
      [[maybe_unused]] auto begin_time = std::chrono::steady_clock::now();
      auto key = key_generator_->NextKey();
      auto value = GenerateValueFromKey(key);

      if (rand_val(rng) < read_prop) {
        s = db_->Get(ReadOptions(), key, &got);
      } else {
        s = db_->Put(WriteOptions(), key, value);
      }

      ++completed_count_;
      if (!s.ok()) {
        ++failed_count_;
      }
    }
  }

  void StatisticsThread() {
    int prev_completed = 0;
    int seconds = 0;
    while (prev_completed < insert_counts_) {
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

  int insert_counts_ = 320000000;

  std::thread* insert_threads_;

  std::thread ops_thread_;

  std::atomic<int> completed_count_{0};

  std::atomic<int> failed_count_{0};

  KeyGenerator* key_generator_ = nullptr;

  float read_prop = 0.0;

  DB* db_;
};

void DoTest(double zipf, double read_ratio) {
  int thread_num = 16;
  int total_count = 320000000;
  int load_count = 80000000;
  uint64_t sample_range = 1000000000ULL;

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;
  options.nvm_path = "/pg_wal/ycc/memory_art";
  options.IncreaseParallelism(16);

  BlockBasedTableOptions table_options;
  table_options.filter_policy.reset(NewBloomFilterPolicy(10));
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  std::string db_path = "/tmp/db_old_custom";

  DB* db;
  DB::Open(options, db_path, &db);

  Inserter load_inserter(thread_num, db);
  load_inserter.SetGenerator(new YCSBLoadGenerator(load_count, sample_range));
  load_inserter.SetInsertCount(load_count);
  load_inserter.DoLoad();

  db->Reset();

  Inserter inserter(thread_num, db);
  inserter.SetReadProp(read_ratio);
  inserter.SetInsertCount(total_count);
  inserter.SetGenerator(
      new YCSBZipfianGenerator(load_count, total_count, sample_range, zipf, 26.4));
  inserter.DoInsert();

  db->Close();

  delete db;
}

int main(int argc, char* argv[]) {
  double zipf = atof(argv[1]);
  double read_ratio = atof(argv[2]);
  DoTest(zipf, read_ratio);
}