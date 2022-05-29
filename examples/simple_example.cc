// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <cstdint>
#include <iostream>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

#include <algorithm>
#include <mutex>
#include <random>
#include <thread>
#include <cmath>
#include <time.h>

using namespace rocksdb;

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

  std::string HighFreqKey(int i, int f) {
    auto ret = gen[i + 1]->highestFreqValue(f);
    std::string s = std::to_string(ret);
    return prefix + std::string(8 - s.length(), '0') + s;
  }

  std::string NextKey() {
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
    std::string s = std::to_string(ret);
    return prefix + std::string(8 - s.length(), '0') + s;
  }

 private:
  std::string prefix = "user";
  std::uniform_int_distribution<int> dist;
  std::default_random_engine rng;
  ScrambledZipfianGenerator* gen[5];
};

///////////////////////////////////////////////////////////////////////////////

int     thread_num = 32;
int     total_count = 134217728;
int     count_per_thread = total_count / thread_num;
int64_t* completed_count = new int64_t[thread_num * 8];

std::string repeat(std::string& str) {
  int repeat_times = 1024UL / str.length();
  size_t len = str.length() * repeat_times;
  std::string ret;
  ret.reserve(1024);
  while (repeat_times--) {
    ret += str;
  }
  ret += str.substr(0, 1024 - ret.length());
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

void PutThread(DB* db, int thread_id) {
  static std::atomic<int64_t> counter{0};
  thread_id *= 8;
  for (int i = 0; i < count_per_thread; ++i) {
    std::string key = "user" + std::to_string(fnvhash64(counter++));
    std::string value = repeat(key);
    db->Put(WriteOptions(), key, value);
    ++completed_count[thread_id];
  }
}

void UpdateThread(DB* db, int thread_id) {
  CustomZipfianGenerator zipf(100000000);
  thread_id *= 8;
  for (int i = 0; i < count_per_thread; ++i) {
    std::string key = zipf.NextKey();
    std::string value = repeat(key);
    db->Put(WriteOptions(), key, value);
    ++completed_count[thread_id];
  }
}

void StatisticsThread() {
  int finished = 0;
  int seconds = 0;
  while (finished < total_count - thread_num) {
    int new_finished = 0;
    for (int i = 0; i < thread_num; ++i) {
      new_finished += completed_count[i * 8];
    }

    int delta = (new_finished - finished) / 5;
    printf("[%d sec] %d operations; %d current ops/sec\n",
           seconds, new_finished, delta);

    finished = new_finished;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    seconds += 5;
  }
  memset(completed_count, 0, thread_num * 64);
}

int main() {
  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  options.OptimizeLevelStyleCompaction();

  memset(completed_count, 0, thread_num * 64);

  DB* db;
  DB::Open(options, "/tmp/db_test", &db);

  std::thread threads[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(UpdateThread, db, i);
  }

  std::thread statistic_thread = std::thread(StatisticsThread);

  for (auto& thread : threads) {
    thread.join();
  }

  statistic_thread.join();

  db->Close();

  delete db;

  return 0;
}