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

namespace ROCKSDB_NAMESPACE {

enum Operation {
  kRead,
  kWrite,
};

void CheckFreq(std::vector<uint64_t>& samples) {
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
}

class KeyGenerator {
 public:
  KeyGenerator(int load_record_count, int run_record_count,
               uint64_t sample_range, double alpha = 0.98)
      : alpha_(alpha),
        interval_length_(sample_range / 100),
        sample_range_(sample_range),
        load_record_count_(load_record_count),
        run_record_count_(run_record_count),
        run_operation_count_(run_record_count) {
    std::random_device rd;
    std::random_device rd2;
    rng_ = std::default_random_engine{rd()};
    rng2_ = std::default_random_engine{rd2()};

    InitializeIntervals();
    Prepare();
  };

  void SetOperationCount(int run_operation_count) {
    if (run_operation_count > run_record_count_) {
      printf("Error: operation count must less than record count\n");
      abort();
    }
    run_operation_count_ = run_operation_count;
  }

  void SetReadRatio(double read_ratio) {
    read_ratio_ = read_ratio;
  }

  void SetPrefix(std::string& prefix) {
    prefix_ = prefix;
  }

  int GetLoadOperations() const {
    return load_record_count_;
  }

  int GetRunOperations() const {
    return run_operation_count_;
  }

  std::string NextLoadKey() {
    return GenerateKeyFromValue(load_records_[load_index_++]);
  }

  Operation Next(std::string& key) {
    if (rand_double(rng_) < read_ratio_) {
      key = NextReadKey();
      return kRead;
    } else {
      key = NextWriteKey();
      return kWrite;
    }
  }

 protected:
  void InitializeIntervals() {
    std::random_device shuffle_rd;
    std::vector<int> r(100);
    std::iota(r.begin(), r.end(), 0);
    std::shuffle(r.begin(), r.end(), shuffle_rd);

    memset(hot_read_intervals_, 0, sizeof(int) * 101);
    memset(hot_write_intervals_, 0, sizeof(int) * 101);
    for (int i = 0; i < 10; ++i) {
      hot_read_intervals_[r[i]] = 1;
      hot_write_intervals_[r[i]] = 1;
    }
    for (int i = 10; i < 20; ++i) {
      hot_read_intervals_[r[i]] = 1;
    }
    for (int i = 20; i < 30; ++i) {
      hot_write_intervals_[r[i]] = 1;
    }
  }

  void Prepare() {
    if (!load_records_.empty()) {
      return;
    }

    // Generate load records
    load_index_.store(0, std::memory_order_relaxed);
    load_records_.resize(load_record_count_);
    for (int i = 0; i < load_record_count_; ++i) {
      load_records_[i] = fnvhash64(i);
    }

    // Generate zipfian data and map them to load records.
    // Extend load records if necessary.
    std::unordered_map<uint64_t, int> frequency = GenerateZipfianData();

    std::vector<uint64_t> hot_data;
    std::vector<uint64_t> cold_data;
    hot_data.reserve(run_records_.size());
    cold_data.reserve(run_records_.size());
    for (auto& pair : frequency) {
      (pair.second == 1 ? cold_data : hot_data).push_back(pair.first);
    }

    std::vector<uint64_t> hot_load_data;
    std::vector<uint64_t> cold_load_data;
    hot_load_data.reserve(run_records_.size());
    cold_load_data.reserve(run_records_.size());
    for (auto& v : load_records_) {
      (hot_write_intervals_[v / interval_length_] ? hot_load_data : cold_load_data).push_back(v);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_int_distribution<uint64_t> dist(0, sample_range_ - 1);

    if (hot_load_data.size() < hot_data.size()) {
      int to_add = hot_data.size() - hot_load_data.size();
      load_records_.reserve(load_records_.size() + to_add);
      for (int i = 0; i < to_add; ++i) {
        uint64_t v = 0;
        do {
          v = dist(rng);
        } while (!hot_write_intervals_[v / interval_length_]);
        hot_load_data.push_back(v);
        load_records_.push_back(v);
      }
    }

    if (cold_load_data.size() < cold_data.size()) {
      int to_add = cold_data.size() - cold_load_data.size();
      load_records_.reserve(load_records_.size() + to_add);
      for (int i = 0; i < to_add; ++i) {
        uint64_t v = 0;
        do {
          v = dist(rng);
        } while (hot_write_intervals_[v / interval_length_]);
        cold_load_data.push_back(v);
        load_records_.push_back(v);
      }
    }

    std::random_device shuffle_rd;
    std::shuffle(hot_load_data.begin(), hot_load_data.end(), shuffle_rd);
    std::shuffle(cold_load_data.begin(), cold_load_data.end(), shuffle_rd);
    std::shuffle(load_records_.begin(), load_records_.end(), shuffle_rd);
    std::unordered_map<uint64_t, uint64_t> mapper;
    int index = 0;
    for (auto v : hot_data) {
      mapper[v] = hot_load_data[index++];
    }
    index = 0;
    for (auto v : cold_data) {
      mapper[v] = cold_load_data[index++];
    }

    for (auto& v : run_records_) {
      v = mapper[v];
    }

    hot_read_records_.reserve(load_records_.size());
    cold_read_records_.reserve(load_records_.size());
    for (auto v : load_records_) {
      (hot_read_intervals_[v / interval_length_] ? hot_read_records_ : cold_read_records_).push_back(v);
    }

    printf("Change load_record_count from %d to %d\n",
           load_record_count_, (int)load_records_.size());
    load_record_count_ = load_records_.size();
  }

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

  std::unordered_map<uint64_t, int> GenerateZipfianData() {
    printf("Start generating zipfian data\n");
    double c = 0.0;

    uint64_t zipfian_sample_range = 1000000000;

    for (size_t i = 1; i <= zipfian_sample_range; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    double* sum_probs = new double[zipfian_sample_range + 1];
    sum_probs[0] = 0;
    for (size_t i = 1; i <= zipfian_sample_range; ++i) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha_);
    }

    std::random_device rd;
    std::default_random_engine rng = std::default_random_engine{rd()};
    std::uniform_real_distribution<> rand_val{0.0, 1.0};

    std::unordered_map<uint64_t, int> frequency;
    run_records_.resize(run_record_count_);
    for (int i = 0; i < run_record_count_; ++i) {
      uint64_t res = 0;
      double z;
      do {
        z = rand_val(rng);
      } while ((z == 0) || (z == 1));

      uint64_t low = 1;
      uint64_t high = zipfian_sample_range;
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

      ++frequency[res];
      run_records_[i] = res;
    }

    return frequency;
  }

  std::string GenerateKeyFromValue(uint64_t value) {
    std::string s = std::to_string(value);
    return prefix_ + std::string(19 - s.length(), '0') + s;
  }

  std::string NextReadKey() {
    static std::uniform_int_distribution<uint64_t> rand1(
        0, hot_read_records_.size() - 1);
    static std::uniform_int_distribution<uint64_t> rand2(
        0, cold_read_records_.size() - 1);

    uint64_t v;
    if (rand_double(rng_) < 0.2) {
      v = hot_read_records_[rand1(rng2_)];
    } else {
      v = cold_read_records_[rand2(rng2_)];
    }

    return GenerateKeyFromValue(v);
  }

  std::string NextWriteKey() {
    return GenerateKeyFromValue(run_records_[update_index_++]);
  }

  double alpha_ = 0.98;

  double read_ratio_ = 0.5;

  int hot_read_intervals_[101];

  int hot_write_intervals_[101];

  uint64_t interval_length_;

  uint64_t sample_range_;

  int load_record_count_;

  int run_record_count_;

  int run_operation_count_;

  std::vector<uint64_t> load_records_;

  std::vector<uint64_t> run_records_;

  std::vector<uint64_t> hot_read_records_;

  std::vector<uint64_t> cold_read_records_;

  std::atomic<int> load_index_;

  std::atomic<int> update_index_;

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng_;

  std::default_random_engine rng2_;

  std::string prefix_ = "user";
};

// Note that whole DBTest and its child classes disable fsync on files
// and directories for speed.
// If fsync needs to be covered in a test, put it in other places.
class DBTest3 : public DBTestBase {
 public:
  DBTest3() : DBTestBase("/db_test3", /*env_do_fsync=*/false) {}
};

std::string repeat(std::string& str) {
  int repeat_times = 1024UL / str.length();
  size_t len = str.length() * repeat_times;
  std::string ret;
  ret.reserve(len);
  while (repeat_times--) {
    ret += str;
  }
  return ret;
}

TEST_F(DBTest3, MockEnvTest) {
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

  auto gen = new KeyGenerator(load_count, run_count, sample_range);
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
