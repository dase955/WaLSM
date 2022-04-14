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

void MultiThreadTest(DB *db, std::vector<std::string> *sampled_keys, int thread_id) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> keyDis(1, 32);
  std::random_device rd2;
  std::mt19937 generator(rd2());

  std::string keyStr("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+[{]}|<,>.?/~`' ");

  std::vector<std::string> sampledKeys;
  sampled_keys->reserve(16384 * 4);
  for (int i = 0; i < 16384 * 64; ++i) {
    std::shuffle(keyStr.begin(), keyStr.end(), generator);
    std::string key = keyStr.substr(0, keyDis(gen));    // assumes 32 < number of characters in str
    std::string value = key + key;

    ASSERT_OK(db->Put(WriteOptions(), key, value));
    if (i % 16 == 0) {
      sampled_keys->push_back(key);
    }
  }
}

TEST_F(DBTest3, MockEnvTest) {
  std::unique_ptr<MockEnv> env{new MockEnv(Env::Default())};
  Options options;
  options.force_consistency_checks = false;
  options.create_if_missing = true;
  options.vlog_file_size = 1072ULL << 20;
  options.vlog_force_gc_ratio_ = 0.25;
  options.compaction_threshold = 256 << 20;
  options.group_split_threshold = 6 << 20;
  options.group_min_size = 2 << 20;
  options.env = env.get();
  DB* db;

  ASSERT_OK(DB::Open(options, "/dir/db", &db));

  std::thread threads[16];
  std::vector<std::string> sampled_keys[16];
  int n = 0;
  for (auto & thread : threads) {
    thread = std::thread(MultiThreadTest, db, &(sampled_keys[n]), n);
    n++;
  }

  for (auto & thread : threads) {
    thread.join();
  }

  std::cout << "Start test get" << std::endl;
  for (auto &vec : sampled_keys) {
    for (auto &key : vec) {
      std::string res;
      std::string expected = key + key;
      auto status = db->Get(ReadOptions(), key, &res);
      //std::cout << key << ": " << res << std::endl;
      ASSERT_TRUE(!status.ok() || res == expected );
      if (!status.ok()) {
        std::cout << "Error!" << std::endl;
      }
    }
  }

  db->Delete(WriteOptions(), sampled_keys[0][0]);
  std::string res;
  auto status = db->Get(ReadOptions(), sampled_keys[0][0], &res);
  assert(status.IsNotFound() && res.empty());

  std::cout << "Test get done" << std::endl;

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
