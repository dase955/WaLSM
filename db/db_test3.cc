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

  long highestFreqValue(int f) {
    return min + fnvhash64(f) % itemcount;
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

class CustomZipfianGenerator {
 public:
  CustomZipfianGenerator(long count) {
    gen[0] = new ScrambledZipfianGenerator(0, count - 1);
    gen[1] = new ScrambledZipfianGenerator(1 * count / 10, 2 * count / 10 - 1);
    gen[2] = new ScrambledZipfianGenerator(2 * count / 10, 3 * count / 10 - 1);
    gen[3] = new ScrambledZipfianGenerator(4 * count / 10, 5 * count / 10 - 1);
    gen[4] = new ScrambledZipfianGenerator(8 * count / 10, 9 * count / 10 - 1);

    std::random_device rd;
    dist = std::uniform_int_distribution<int>{1, 100};
    rng = std::default_random_engine{rd()};
  }

  ~CustomZipfianGenerator() {
    for (auto& g : gen) {
      delete g;
    }
  }

  long NextKey() {
    int r = dist(rng);
    long ret = 0;
    if (r <= 10) {
      ret = gen[0]->nextValue();
    } else if (r <= 32) {
      ret = gen[1]->nextValue();
    } else if (r <= 54) {
      ret = gen[2]->nextValue();
    } else if (r <= 77) {
      ret = gen[3]->nextValue();
    } else {
      ret = gen[4]->nextValue();
    }
    return ret;
  }

 private:
  std::uniform_int_distribution<int> dist;
  std::default_random_engine rng;
  ScrambledZipfianGenerator* gen[5];
};

// Note that whole DBTest and its child classes disable fsync on files
// and directories for speed.
// If fsync needs to be covered in a test, put it in other places.
class DBTest3 : public DBTestBase {
 public:
  DBTest3() : DBTestBase("/db_test3", /*env_do_fsync=*/false) {}
};

int thread_num = 16;
int total_count = 16384 * 8192;
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

std::atomic<int64_t> counter{0};

std::vector<int> final_samples;

void Test1Thread(DB *db) {
  for (int i = 0; i < count_per_thread; ++i) {
    auto keynum = fnvhash64(counter++);
    std::string key = "user" + std::to_string(keynum);
    std::string value = repeat(key);
    ASSERT_OK(db->Put(WriteOptions(), key, value));
  }
}

void Test3Thread(DB* db, std::unordered_set<long>& map) {
  CustomZipfianGenerator zipf(100000000);
  std::string prefix = "user";
  for (int i = 0; i < count_per_thread; ++i) {
    long ret = zipf.NextKey();
    std::string s = std::to_string(ret);
    std::string key = prefix + std::string(8 - s.length(), '0') + s;
    std::string value = repeat(key);
    db->Put(WriteOptions(), key, value);
    if (i % 16 == 0) {
      map.emplace(ret);
    }
  }
}

void Test3(DB* db) {
  std::thread threads[thread_num];
  std::unordered_set<long> maps[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(Test3Thread, db, std::ref(maps[i]));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::cout << "Start testing" << std::endl;
  for (auto& map : maps) {
    std::string prefix = "user";
    std::cout << "Test num: " << map.size() << std::endl;
    for (auto& num : map) {
      std::string s = std::to_string(num);
      std::string key = prefix + std::string(8 - s.length(), '0') + s;
      std::string res;
      std::string expected = repeat(key);
      auto status = db->Get(ReadOptions(), key, &res);
      ASSERT_TRUE(!status.ok() || res == expected);
      if (!status.ok()) {
        std::cout << key << " Error!" << std::endl;
      }
    }
  }
}

void Test4Thread(DB* db) {
  std::string prefix = "user";
  for (int i = 0; i < count_per_thread; ++i) {
    int val = final_samples[counter++];
    std::string s = std::to_string(val);
    std::string key = prefix + std::string(9 - s.length(), '0') + s;
    std::string value = repeat(key);
    db->Put(WriteOptions(), key, value);
  }
}

void Test4(DB* db) {
  std::thread threads[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(Test4Thread, db);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

int      sample_range = 1000000000;

void GenerateSamples() {
  final_samples.resize(total_count);
  std::uniform_int_distribution<int> dist(0, sample_range - 1);
  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};

  int cold_count = 0;
  int hot_count;

  {
    ZipfianGenerator gen(0, sample_range, 0.99, 26.46902820178302);
    std::unordered_map<int, int> freqs;
    int left[2] = {400000000, 800000000};

    for (int i = 0; i < total_count; ++i) {
      long value = gen.nextValue();
      freqs[value]++;
      final_samples[i] = (int)value;
    }

    for (auto& value : final_samples) {
      cold_count += (freqs[value] == 1);
      value = freqs[value] == 1
                  ? dist(rng) :
                  (int)(fnvhash64(value) % 50000000) + left[value % 2];
    }

    hot_count = freqs.size() - cold_count;
  }

  float hot_freq = (total_count - cold_count) / (float)total_count * 100.f;
  printf("Sample generated. hot count = %d(%.2f), cold count = %d\n",
         hot_count, hot_freq, cold_count);
}

TEST_F(DBTest3, MockEnvTest) {

  GenerateSamples();

  return;

  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  options.OptimizeLevelStyleCompaction();

  int sample_size = 160000000;

  final_samples.reserve(sample_size);
  std::uniform_int_distribution<int> dist(0, 1000000000 - 1);
  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};

  {
    ZipfianGenerator gen(0, 1000000000L, 0.99, 26.46902820178302);
    std::unordered_map<int, int> freqs;
    int left[2] = {400000000, 800000000};

    for (int i = 0; i < sample_size; ++i) {
      long value = gen.nextValue();
      assert(value <= 1000000000L);
      freqs[value]++;
      final_samples[i] = (int)value;
    }

    for (auto& value : final_samples) {
      value = freqs[value] == 1
                  ? dist(rng) :
                  (int)(fnvhash64(value) % 50000000) + left[value % 2];
    }
  }

  DB* db;

  ASSERT_OK(DB::Open(options, "/tmp/db_test", &db));

  Test4(db);

  db->Close();

  delete db;
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
