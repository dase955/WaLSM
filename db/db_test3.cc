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

class ZipfianGenerator {
 public:
  ZipfianGenerator(long min, long max, double zipfianconstant_, double zetan_) {
    items = max - min + 1;
    base = min;
    this->zipfianconstant = zipfianconstant_;

    theta = this->zipfianconstant;

    zeta2theta = zeta(2, theta);

    alpha = 1.0 / (1.0 - theta);
    this->zetan = zetan_;
    eta = (1 - std::pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / this->zetan);

    std::random_device rd;
    rng = std::default_random_engine{rd()};

    nextValue();
  }

  double zeta(long n, double thetaVal) {
    return zetastatic(n, thetaVal);
  }

  static double zetastatic(long n, double theta) {
    return zetastatic(0, n, theta, 0);
  }

  static double zetastatic(long st, long n, double theta, double initialsum) {
    double sum = initialsum;
    for (long i = st; i < n; i++) {

      sum += 1 / (std::pow(i + 1, theta));
    }

    return sum;
  }

  long nextLong(long itemcount) {
    double u = rand_double(rng);
    double uz = u * zetan;

    if (uz < 1.0) {
      return base;
    }

    if (uz < 1.0 + std::pow(0.5, theta)) {
      return base + 1;
    }

    return base + (long) ((itemcount) * std::pow(eta * u - eta + 1, alpha));
  }

  long nextValue() {
    return nextLong(items);
  }

 private:
  long items;

  long base;

  double zipfianconstant;

  double alpha, zetan, eta, theta, zeta2theta;

  std::uniform_real_distribution<> rand_double{0.0, 1.0};

  std::default_random_engine rng;
};

class ScrambledZipfianGenerator {
 public:
  ScrambledZipfianGenerator(long min_, long max_) {
    this->min = min_;
    this->max = max_;
    itemcount = this->max - this->min + 1;
    gen = new ZipfianGenerator(0, ITEM_COUNT, 0.99, ZETAN);
  }

  ~ScrambledZipfianGenerator() {
    delete gen;
  }

  long nextValue() {
    long ret = gen->nextValue();
    return min + fnvhash64(ret) % itemcount;
  }

 private:
  static long fnvhash64(long val) {
    //from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
    long hashval = 0xCBF29CE484222325L;

    for (int i = 0; i < 8; i++) {
      long octet = val & 0x00ff;
      val = val >> 8;

      hashval = hashval ^ octet;
      hashval = hashval * 1099511628211L;
      //hashval = hashval ^ octet;
    }
    return std::fabs(hashval);
  }

  double ZETAN = 26.46902820178302;
  long ITEM_COUNT = 10000000000L;

  ZipfianGenerator* gen;
  long min, max, itemcount;
};

int thread_num = 8;
int total_count = 40000000;
int count_per_thread = total_count / thread_num;
std::atomic<int64_t> counter{0};
std::vector<uint64_t> written(total_count);

std::string NumToKey(uint64_t key_num) {
  std::string s = std::to_string(key_num);
  return "user" + s;
}

std::string GenerateValueFromKey(std::string& key) {
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

unsigned long fnvhash64(int64_t val) {
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

void ZipfianPutThread(DB *db) {
  ScrambledZipfianGenerator zipf(0, 1000000000);
  for (int i = 0; i < count_per_thread; ++i) {
    auto key_num = zipf.nextValue();
    std::string key = NumToKey(key_num);
    std::string value = GenerateValueFromKey(key);
    assert(db->Put(WriteOptions(), key, value).ok());
    written[counter.fetch_add(1, std::memory_order_relaxed)] = key_num;
  }
}

void UniformPutThread(DB *db) {
  for (int i = 0; i < count_per_thread; ++i) {
    auto c = counter++;
    auto key_num = fnvhash64(c);
    std::string key = NumToKey(key_num);
    std::string value = GenerateValueFromKey(key);
    assert(db->Put(WriteOptions(), key, value).ok());
    written[c] = key_num;
  }
}

void GetThread(DB *db) {
  std::random_device rd;
  std::default_random_engine rng{rd()};

  while (true) {
    int cur_counter = counter.load(std::memory_order_relaxed);
    if (cur_counter > total_count - 128) {
      break;
    }

    if (cur_counter <= 128) {
      continue;
    }

    std::uniform_int_distribution<long> dist {1, cur_counter - 128};
    auto c = dist(rng);

    std::string res;
    std::string key = NumToKey(written[c]);
    std::string expected = GenerateValueFromKey(key);

    auto status = db->Get(ReadOptions(), key, &res);
    assert(!status.ok() || res == expected);
    if (!status.ok()) {
      printf("%s Error\n", key.c_str());
    }
  }
}

std::atomic<int64_t> check_counter{0};

void CheckThread(DB* db) {
  for (int i = 0; i < count_per_thread; ++i) {
    auto c = check_counter++;

    std::string res;
    std::string key = NumToKey(written[c]);
    std::string expected = GenerateValueFromKey(key);

    auto status = db->Get(ReadOptions(), key, &res);
    assert(!status.ok() || res == expected);
    if (!status.ok()) {
      std::cout << key << " Error!" << std::endl;
    }
  }
}

void DoTest(std::string test_name) {
  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.nvm_path = "/tmp/nodememory";
  options.compression = rocksdb::kNoCompression;
  options.IncreaseParallelism(16);

  DB* db;
  DB::Open(options, "/tmp/tmp_data/db_test_art", &db);

  std::thread read_threads[thread_num];
  std::thread write_threads[thread_num];
  std::thread check_threads[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    write_threads[i] = std::thread(ZipfianPutThread, db);
    read_threads[i] = std::thread(GetThread, db);
  }

  for (auto & thread : read_threads) {
    thread.join();
  }

  for (auto & thread : write_threads) {
    thread.join();
  }


  auto iter = db->NewIterator(ReadOptions());

  std::string key = NumToKey(written[total_count / 10]);
  iter->Seek(key);

  for (int i = 0; i < 100 && iter->Valid(); ++i) {
    auto k = iter->key();
    std::string kk(k.data(), k.size());
    std::cout << kk << std::endl;
    iter->Next();
  }

  delete iter;

  //return;

  //db->Reset();

  std::cout << "Start test get" << std::endl;

  for (int i = 0; i < thread_num; ++i) {
    check_threads[i] = std::thread(CheckThread, db);
  }

  for (int i = 0; i < thread_num; ++i) {
    check_threads[i].join();
  }

  delete db;
}
// Note that whole DBTest and its child classes disable fsync on files
// and directories for speed.
// If fsync needs to be covered in a test, put it in other places.
class DBTest3 : public DBTestBase {
 public:
  DBTest3() : DBTestBase("/db_test3", /*env_do_fsync=*/false) {}
};

TEST_F(DBTest3, MockEnvTest) {
  setbuf(stdout, NULL);
  DoTest("art");
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
