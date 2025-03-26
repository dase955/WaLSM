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

class KeyGenerator {
 public:
  KeyGenerator(int record_count, int sample_range)
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
  std::string NextKey() {
    std::string s = std::to_string(Next());
    return prefix_ + s;
  }

 protected:
  int record_count_;

  int sample_range_;

  std::string prefix_ = "user";
};

class YCSBZipfianGenerator : public KeyGenerator{
 public:
  YCSBZipfianGenerator(int record_count, int sample_range, double theta, double zetan)
      : KeyGenerator(record_count, sample_range), theta_(theta), zetan_(zetan) {
    alpha = 1.0 / (1.0 - theta);
    eta = (1 - std::pow(2.0 / sample_range_, 1 - theta))
          / (1 - zetastatic(2, theta) / zetan_);

    std::random_device rd;
    rng = std::default_random_engine{rd()};
  }

  static double zetastatic(long n, double theta) {
    double sum = 0.0f;
    for (long i = 0; i < n; i++) {
      sum += 1 / (std::pow(i + 1, theta));
    }
    return sum;
  }

  uint64_t Next() {
    double u = rand_double(rng);
    double uz = u * zetan_;

    if (uz < 1.0) {
      return fnvhash64(0);
    }

    if (uz < 1.0 + std::pow(0.5, theta_)) {
      return fnvhash64(1);
    }

    int v = (int) ((sample_range_) * std::pow(eta * u - eta + 1, alpha));
    return fnvhash64(v);
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

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng;
};

class CustomZipfianGenerator : public KeyGenerator {
 public:
  CustomZipfianGenerator(int record_count, int sample_range, double alpha)
      : KeyGenerator(record_count, sample_range), alpha_(alpha) {
    records_.resize(record_count);
  }

  void Prepare() override {
    double c = 0.0;
    for (int i = 1; i <= sample_range_; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    double* sum_probs = new double[sample_range_ + 1];
    sum_probs[0] = 0;
    for (int i = 1; i <= sample_range_; ++i) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha_);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_real_distribution<> rand_val{0.0, 1.0};

    printf("Generate samples\n");

    std::unordered_map<int, int> freqs;
    for (int i = 0; i < record_count_; ++i) {
      long res = 0;
      double z;
      do {
        z = rand_val(rng);
      } while ((z == 0) || (z == 1));

      int low = 1;
      int high = sample_range_;
      int mid;
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
      records_[i] = res;
    }

    std::unordered_map<int, int> modified;
    ShuffleSamples(freqs, modified, sample_range_);
    for (auto& value : records_) {
      value = modified[value];
    }

    delete[] sum_probs;

    CheckFreq(records_);
  }

  uint64_t Next() override {
    return records_[cur_index_++];
  }

 private:
  static void ShuffleSamples(
      std::unordered_map<int, int>& freqs,
      std::unordered_map<int, int>& modified,
      int sample_range) {
    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_int_distribution<int> dist(0, sample_range - 1);

    int left[2] = {400000000, 800000000};
    int interval = 50000000;
    int mod = interval * 2;

    std::random_device shuffle_rd;
    std::vector<int> shuffled(mod);
    std::iota(std::begin(shuffled), std::end(shuffled), 0);
    std::shuffle(shuffled.begin(), shuffled.end(), shuffle_rd);

    auto check_func = [&](int val) {
      for (auto l : left) {
        if (val >= l && val <= l + interval) {
          return false;
        }
      }
      return true;
    };

    int cur = 0;
    for (auto& pair : freqs) {
      int old_value = pair.first;
      if (pair.second == 1) {
        int new_value = dist(rng);
        while (!check_func(new_value)) {
          new_value = dist(rng);
        }
        modified[old_value] = new_value;
      } else {
        int new_value = shuffled[cur++];
        modified[old_value] = left[new_value / interval] + (new_value % interval);
      }
    }
  }

  static void CheckFreq(std::vector<int>& samples) {
    std::unordered_map<int, int> freqs;
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

  double alpha_;

  std::vector<int> records_;

  std::atomic<int> cur_index_{0};
};

class YCSBLoadGenerator : public KeyGenerator {
 public:
  YCSBLoadGenerator(int record_count, int sample_range)
      : KeyGenerator(record_count, sample_range) {};

  uint64_t Next() override {
    return fnvhash64(cur_index_++);
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

  std::atomic<int> cur_index_{0};
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

  void SetMetricInterval(int interval) {
    interval_ = interval;
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
  void Insert(int thread_id) {
    int insert_per_thread = insert_counts_ / num_threads_;
    if (thread_id == 0) {
      insert_per_thread += (insert_counts_ % num_threads_);
    }

    while (insert_per_thread--) {
      [[maybe_unused]] auto begin_time = std::chrono::steady_clock::now();
      auto key = key_generator_->NextKey();
      auto value = GenerateValueFromKey(key);
      auto status = db_->Put(WriteOptions(), key, value);
      [[maybe_unused]] auto end_time = std::chrono::steady_clock::now();

      ++completed_count_;
      if (!status.ok()) {
        assert(false);
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

  DB* db_;
};

void ParseOptions(Options& options) {
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  // options.OptimizeLevelStyleCompaction();
  options.OptimizeUniversalStyleCompaction();

  std::ifstream option_file("options.txt", std::ios::in);
  std::string line;
  while (getline(option_file, line)) {
    if (line.substr(0, 16) == "timestamp_factor") {
      options.timestamp_factor = std::atoi(line.substr(17).c_str());
    } else if (line.substr(0, 17) == "layer_ts_interval") {
      options.layer_ts_interval = std::atoi(line.substr(18).c_str());
      options.timestamp_waterline = options.layer_ts_interval * 10;
    } else if (line.substr(0, 14) == "group_min_size") {
      options.group_min_size = std::atoi(line.substr(15).c_str()) << 10;
    } else if (line.substr(0, 21) == "group_split_threshold") {
      options.group_split_threshold = std::atoi(line.substr(22).c_str()) << 20;
    }
  }

  printf("Parse option done.\n");
  printf("options.timestamp_factor = %d\n", options.timestamp_factor);
  printf("options.layer_ts_interval = %d\n", options.layer_ts_interval);
  printf("options.timestamp_waterline = %d\n", options.timestamp_waterline);
  printf("options.group_min_size = %.2fk\n", options.group_min_size / 1048576.f);
  printf("options.group_split_threshold = %dM\n",
         options.group_split_threshold / 1048576);
}

void DoTest(std::string test_name) {
  int thread_num = 8;
  int total_count = 320000000;
  int sample_range = 1000000000;

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;
  options.nvm_path = "/mnt/walsm/node_memory";
  options.IncreaseParallelism(16);

  std::string db_path = "/tmp/tmp_data/db_test_" + test_name;

  DB* db;
  DB::Open(options, db_path, &db);

  Inserter inserter(thread_num, db);
  inserter.SetGenerator(
      new YCSBZipfianGenerator(total_count, sample_range, 0.98, 26.49));
  inserter.DoInsert();

  db->Close();

  delete db;
}

int main(int argc, char* argv[]) {
  DoTest("art");

  return 0;
}