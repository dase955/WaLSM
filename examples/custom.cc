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

    std::random_device rd;
    rng = std::default_random_engine{rd()};

    std::cout << "theta = " << theta << ", zetan = " << zetan_ << std::endl;
  }

  static double zetastatic(long n, double theta) {
    double sum = 0.0f;
    long t = n / 100;
    for (long i = 0; i < n; i++) {
      if (t && i % t == 0) {
        std::cout << i << std::endl;
      }
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
      std::random_device shuffle_rd;
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

  uint64_t Next() {
    return std::hash<uint64_t>{}(samples_[index_++]);
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

  std::string NextKey() override {
    std::string s = std::to_string(Next());
    return prefix_ + std::string(10 - s.length(), '0') + s;
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

  int interval_ = 60;

  int num_threads_ = 16;

  int insert_counts_ = 320000000;

  std::thread* insert_threads_;

  std::thread ops_thread_;

  std::atomic<int> completed_count_{0};

  std::atomic<int> failed_count_{0};

  KeyGenerator* key_generator_ = nullptr;

  DB* db_;
};

void test() {
  int thread_num = 8;
  int total_count = 320000000;
  int sample_range = 1000000000;
  auto gen = new YCSBZipfianGenerator(100000000, total_count, sample_range, 0.98, 26.469);
  gen->Prepare();


  std::unordered_map<uint64_t, int> freqs;
  for (int i = 0; i < total_count; ++i) {
    freqs[gen->Next()]++;
  }

  std::vector<std::pair<uint64_t, int>> pairs;
  for (auto& pair : freqs) {
    pairs.push_back(pair);
  }

  std::sort(pairs.begin(), pairs.end(), [](const std::pair<uint64_t, int>& p1, const std::pair<uint64_t, int>& p2){return p1.second > p2.second;} );

  int acc = 0;
  int sample_count = pairs.size();
  for (int i = 1; i <= 100; ++i) {
    int start = sample_count / 100 * (i - 1);
    int end =   sample_count / 100 * i;
    int sum = 0;
    if (i == 100) {
      end = sample_count;
    }

    for (int j = start; j < end; ++j) {
      sum += pairs[j].second;
    }
    acc += sum;
    float data_size = (float)acc * (1024 + 22 + 8 + 3) / 1024.0 / 1024.0 / 1024.0;
    std::cout << i << "%, " << acc << ", " << (float)acc / total_count << ", "  << data_size << "G" << std::endl;
  }
}


void DoTest(double zipf) {
  int thread_num = 8;
  int total_count = 320000000;
  int load_count = 100000000;
  uint64_t sample_range = 1000000000ULL;

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;
  options.nvm_path = "/pg_wal/ycc/memory_art";
  options.IncreaseParallelism(16);

  std::string db_path = "/tmp/db_old_custom";

  DB* db;
  DB::Open(options, db_path, &db);

  Inserter inserter(thread_num, db);
  inserter.SetGenerator(
      new CustomZipfianGenerator(total_count, sample_range, zipf));
  inserter.DoInsert();

  db->Close();

  delete db;
}

int main(int argc, char* argv[]) {
  double zipf = atof(argv[1]);
  DoTest(zipf);
}
