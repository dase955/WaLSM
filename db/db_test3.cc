//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// Introduction of SyncPoint effectively disabled building and running this test
// in Release build.
// which is a pity, it is a good test
#include <fcntl.h>
#include <algorithm>
#include <set>
#include <thread>
#include <unordered_set>
#include <utility>
#ifndef OS_WIN
#include <unistd.h>
#endif
#ifdef OS_SOLARIS
#include <alloca.h>
#endif

#include "cache/lru_cache.h"
#include "db/blob/blob_index.h"
#include "db/db_impl/db_impl.h"
#include "db/db_test_util.h"
#include "db/dbformat.h"
#include "db/job_context.h"
#include "db/version_set.h"
#include "db/write_batch_internal.h"
#include "env/mock_env.h"
#include "file/filename.h"
#include "memtable/hash_linklist_rep.h"
#include "monitoring/thread_status_util.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/cache.h"
#include "rocksdb/compaction_filter.h"
#include "rocksdb/convenience.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/experimental.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/perf_context.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/snapshot.h"
#include "rocksdb/table.h"
#include "rocksdb/table_properties.h"
#include "rocksdb/thread_status.h"
#include "rocksdb/utilities/checkpoint.h"
#include "rocksdb/utilities/optimistic_transaction_db.h"
#include "rocksdb/utilities/write_batch_with_index.h"
#include "table/mock_table.h"
#include "table/scoped_arena_iterator.h"
#include "test_util/sync_point.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"
#include "util/compression.h"
#include "util/mutexlock.h"
#include "util/random.h"
#include "util/rate_limiter.h"
#include "util/string_util.h"
#include "utilities/merge_operators.h"

#include <cstdio>
#include <string>
#include <cstdint>

#include <time.h>
#include <mutex>
#include <random>
#include <cmath>

namespace ROCKSDB_NAMESPACE {

// Note that whole DBTest and its child classes disable fsync on files
// and directories for speed.
// If fsync needs to be covered in a test, put it in other places.
class DBTest3 : public DBTestBase {
 public:
  DBTest3() : DBTestBase("/db_test3", /*env_do_fsync=*/false) {}
};

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

    std::cout << "theta = " << theta << ", zetan = " << zetan << std::endl;
  }

  static double zetastatic(long n, double theta) {
    double sum = 0.0f;
    for (long i = 0; i < n; i++) {
      sum += 1 / (std::pow(i + 1, theta));
    }
    return sum;
  }

  void Prepare() override {
    samples_.resize(record_count_);
    std::unordered_map<uint64_t, int> frequency;

    for (int i = 0; i < record_count_; ++i) {
      samples_[i] = Zipf();
      ++frequency[samples_[i]];
    }

    printf("Distinct count = %zu\n", frequency.size());
    CheckFreq(samples_);

    // Sort samples by frequency to avoid collision of high frequency data.
    std::unordered_map<uint64_t, uint64_t> mapper;
    std::vector<std::pair<int, uint64_t>> frequency_sorted;
    frequency_sorted.reserve(frequency.size());
    for (auto& pair : frequency) {
      frequency_sorted.emplace_back(pair.second, pair.first);
    }
    std::sort(frequency_sorted.begin(), frequency_sorted.end());

    size_t runs = (frequency.size() + load_count_ - 1) / load_count_;
    for (size_t run = 0; run < runs; ++run) {
      std::random_device shuffle_rd;
      std::vector<int> shuffled(load_count_);
      std::iota(std::begin(shuffled), std::end(shuffled), 0);
      std::shuffle(shuffled.begin(), shuffled.end(), shuffle_rd);

      for (size_t i = 0; i < load_count_ && i < frequency.size(); ++i) {
        mapper[frequency_sorted[i + run * load_count_].second] = shuffled[i];
      }
    }

    for (auto& sample : samples_) {
      sample = mapper[sample];
    }
    CheckFreq(samples_);
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
    return std::hash<uint64_t>{}(Zipf() % 80000000);
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
    for (uint64_t i = 1; i <= sample_range_; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    double* sum_probs = new double[sample_range_ + 1];
    sum_probs[0] = 0;
    for (uint64_t i = 1; i <= sample_range_; ++i) {
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

std::atomic<int> counter{0};
std::atomic<int> failed_counter{0};

std::vector<uint64_t> values;
std::vector<uint64_t> failed_values;

int insert_times = 20000000;

void DoInsert(DB* db, KeyGenerator* gen) {
  int thread_num = 8;
  int thread_count = insert_times / thread_num;
  for (int i = 0; i < thread_count; ++i) {
    auto v = gen->Next();
    std::string key = "user" + std::to_string(v);
    std::string value = Inserter::GenerateValueFromKey(key);
    ASSERT_OK(db->Put(WriteOptions(), key, value));
    values[counter++] = v;
  }
}

void DoGet(DB* db) {
  while (counter.load(std::memory_order_acquire) < insert_times - 16) {
    auto c = counter.load(std::memory_order_acquire);
    if (c < 16) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      continue;
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_int_distribution<int> dist(0, c - 1);

    auto index = dist(rng);
    auto v = values[index];
    std::string key = "user" + std::to_string(v);
    std::string value;
    std::string expected = Inserter::GenerateValueFromKey(key);
    auto status = db->Get(ReadOptions(), key, &value);
    if (status.IsNotFound()) {
      printf("Error not found\n");
    }
  }
}

void DoTest2() {
  setbuf(stdout, nullptr);

  int thread_num = 8;
  int total_count = 320000000;
  uint64_t sample_range = 1000000000UL;

  values.reserve(total_count);
  failed_values.reserve(total_count);
  auto gen = new YCSBZipfianGenerator(80000000, total_count, sample_range, 0.98, 26.469028);

  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.compression = rocksdb::kNoCompression;
  options.nvm_path = "/mnt/chen/vlog";
  options.IncreaseParallelism(16);

  std::string db_path = "/tmp/db_test";

  DB* db;
  DB::Open(options, db_path, &db);

  std::vector<std::thread> put_threads(thread_num);
  std::vector<std::thread> get_threads(thread_num);
  for (auto& thread : put_threads) {
    thread = std::thread(DoInsert, db, gen);
  }
  for (auto& thread : get_threads) {
    thread = std::thread(DoGet, db);
  }
  for (auto& thread : put_threads) {
    thread.join();
  }
  for (auto& thread : get_threads) {
    thread.join();
  }

  for (int i = 0; i < insert_times; ++i) {
    auto v = values[i];
    std::string key = "user" + std::to_string(v);
    std::string value;
    std::string expected = Inserter::GenerateValueFromKey(key);
    auto status = db->Get(ReadOptions(), key, &value);
    if (status.IsNotFound()) {
      printf("Error not found again %s\n", key.c_str());
      status = db->Get(ReadOptions(), key, &value);
    }
  }

  db->Close();
}

TEST_F(DBTest3, MockEnvTest) {
  uint8_t fingerprints[32];
  for (int i = 0; i < 32; ++i) {
    fingerprints[i] = i * 2 + 1;
  }

  fingerprints[1] = fingerprints[4] = fingerprints[8] = fingerprints[13] = 52;

  __m256i target = _mm256_set1_epi8(52);
  __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(fingerprints));
  __m256i r = _mm256_cmpeq_epi8(f, target);
  auto res = (unsigned int)_mm256_movemask_epi8(r);
  while (res) {
    int t = __builtin_ctz(res);
    std::cout << __builtin_ctz(res) << std::endl;

  }
  return;

  DoTest2();
}

/*
TEST_F(DBTest3, MockEnvTest2) {
  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;

  options.compaction_threshold = 1024 << 20;
  options.vlog_force_gc_ratio_ = 0.5;
  options.OptimizeLevelStyleCompaction();

  DB* db;

  ASSERT_OK(DB::Open(options, "/tmp/db_test", &db));

  std::cout << "Start test get" << std::endl;
  for (int i = 0; i < total_count; i += 4) {
    std::string key = next_key(i);
    std::string res;
    std::string expected = repeat(key);
    auto status = db->Get(ReadOptions(), key, &res);
    ASSERT_TRUE(!status.ok() || res == expected );
    if (!status.ok()) {
      std::cout << key << " Error!" << std::endl;
    }
  }
  std::cout << "Test get done" << std::endl;

  db->Close();
  delete db;
}

TEST_F(DBTest3, MockEnvTest3) {
  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;

  options.compaction_threshold = 1024 << 20;
  options.vlog_force_gc_ratio_ = 0.5;
  options.OptimizeLevelStyleCompaction();

  DB* db;

  ASSERT_OK(DB::Open(options, "/tmp/db_test", &db));

  std::thread read_threads[thread_num];
  std::thread write_threads[thread_num];
  std::vector<std::string> sampled_keys[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    write_threads[i] = std::thread(PutThread, db);
    read_threads[i] = std::thread(GetThread, db);
  }

  for (auto & thread : read_threads) {
    thread.join();
  }

  for (auto & thread : write_threads) {
    thread.join();
  }

  std::cout << "Start test get" << std::endl;
  for (int i = 0; i < total_count; i += 4) {
    std::string key = next_key(i);
    std::string res;
    std::string expected = repeat(key);
    auto status = db->Get(ReadOptions(), key, &res);
    ASSERT_TRUE(!status.ok() || res == expected );
    if (!status.ok()) {
      std::cout << key << " Error!" << std::endl;
    }
  }
  std::cout << "Test get done" << std::endl;

  db->Close();

  delete db;
}
*/

}  // namespace ROCKSDB_NAMESPACE

#ifdef ROCKSDB_UNITTESTS_WITH_CUSTOM_OBJECTS_FROM_STATIC_LIBS
extern "C" {
void RegisterCustomObjects(int argc, char** argv);
}
#else
void RegisterCustomObjects(int /*argc*/, char** /*argv*/) {}
#endif  // !ROCKSDB_UNITTESTS_WITH_CUSTOM_OBJECTS_FROM_STATIC_LIBS

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
