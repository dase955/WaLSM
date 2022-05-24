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

// Note that whole DBTest and its child classes disable fsync on files
// and directories for speed.
// If fsync needs to be covered in a test, put it in other places.
class DBTest3 : public DBTestBase {
 public:
  DBTest3() : DBTestBase("/db_test3", /*env_do_fsync=*/false) {}
};

int thread_num = 16;
int total_count = 16384 * 1024;
int count_per_thread = total_count / thread_num;

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

static int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325LL;
static int64_t FNV_PRIME_64 = 1099511628211L;

unsigned long fnvhash64(int64_t val) {
  //from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
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

int* arr = new int[total_count];
std::atomic<int64_t> counter{0};

std::string next_key(int64_t c) {
  auto keynum = fnvhash64(c);
  std::string key = std::to_string(keynum);
  return "user" + key;
}

void PutThread(DB *db) {
  for (int i = 0; i < count_per_thread; ++i) {
    int64_t c = counter++;
    assert(arr[c] == 0);
    auto keynum = fnvhash64(c);
    std::string key = "user" + std::to_string(keynum);
    std::string value = repeat(key);
    ASSERT_OK(db->Put(WriteOptions(), key, value));
    arr[c] = 1;
  }
}

void GetThread(DB *db) {
  while (true) {
    int cur_counter = counter.load(std::memory_order_relaxed);
    if (cur_counter > total_count - 10000) {
      break;
    }

    if (cur_counter <= 50000) {
      continue;
    }

    std::uniform_int_distribution<long> dist {1, cur_counter - 50000};
    std::random_device rd;
    std::default_random_engine rng {rd()};

    auto c = dist(rng);
    assert(arr[c] == 1);
    std::string key = next_key(c);
    std::string res;
    std::string expected = repeat(key);
    auto status = db->Get(ReadOptions(), key, &res);
    ASSERT_TRUE(!status.ok() || res == expected);
    if (!status.ok()) {
      status = db->Get(ReadOptions(), key, &res);
      std::cout << key << " Error!" << std::endl;
    }
  }
}

TEST_F(DBTest3, MockEnvTest) {
  memset(arr, 0, sizeof(int) * total_count);

  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;

  options.compaction_threshold = 1024 << 20;
  options.vlog_force_gc_ratio_ = 0.5;
  options.OptimizeLevelStyleCompaction();

  DB* db;

  ASSERT_OK(DB::Open(options, "/tmp/db_test", &db));

  std::thread threads[thread_num];
  std::vector<std::string> sampled_keys[thread_num];
  for (auto& thread : threads) {
    thread = std::thread(PutThread, db);
  }

  for (auto & thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(std::chrono::seconds(30));

  std::cout << "Start test get" << std::endl;
  for (int i = 0; i < total_count; i += 2) {
    std::string key = next_key(i);
    std::string res;
    std::string expected = repeat(key);
    auto status = db->Get(ReadOptions(), key, &res);
    ASSERT_TRUE(!status.ok() || res == expected );
    if (!status.ok()) {
      status = db->Get(ReadOptions(), key, &res);
      std::cout << key << " Error!" << std::endl;
    }
  }
  std::cout << "Test get done" << std::endl;

  db->Close();

  delete db;
}

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
