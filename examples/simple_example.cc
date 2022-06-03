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

///////////////////////////////////////////////////////////////////////////////

int      thread_num = 32;
int      total_count = 160000000;
int      sample_range = 1000000000;
int      count_per_thread = total_count / thread_num;
int64_t* completed_count = new int64_t[thread_num * 8];
std::vector<int> final_samples;
std::atomic<int64_t> counter{0};

std::string GenerateValue(std::string& str) {
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

void Test1Thread(DB* db, int thread_id) {
  static std::atomic<int64_t> counter{0};
  thread_id *= 8;
  for (int i = 0; i < count_per_thread; ++i) {
    auto keynum = fnvhash64(counter++);
    std::string key = "user" + std::to_string(keynum);
    std::string value = GenerateValue(key);
    assert(db->Put(WriteOptions(), key, value).ok());
    ++completed_count[thread_id];
  }
}

// Default YCSB zipfian workload
void Test2Thread(DB* db, int thread_id) {
  ScrambledZipfianGenerator zipf(0, total_count);
  thread_id *= 8;
  for (int i = 0; i < count_per_thread; ++i) {
    long ret = zipf.nextValue();
    std::string key = "user" + std::to_string(fnvhash64(ret));
    std::string value = GenerateValue(key);
    db->Put(WriteOptions(), key, value);
    ++completed_count[thread_id];
  }
}

void Test3Thread(DB* db, int thread_id) {
  thread_id *= 8;
  std::string prefix = "user";
  for (int i = 0; i < count_per_thread; ++i) {
    int val = final_samples[counter++];
    std::string s = std::to_string(val);
    std::string key = prefix + std::string(9 - s.length(), '0') + s;
    std::string value = GenerateValue(key);
    db->Put(WriteOptions(), key, value);
    ++completed_count[thread_id];
  }
}

void UniformThread(DB* db, int thread_id) {
  std::uniform_int_distribution<int> dist(0, sample_range - 1);
  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};

  thread_id *= 8;
  std::string prefix = "user";

  for (int i = 0; i < count_per_thread; ++i) {
    int val = dist(rng);
    std::string s = std::to_string(val);
    std::string key = prefix + std::string(9 - s.length(), '0') + s;
    std::string value = GenerateValue(key);
    db->Put(WriteOptions(), key, value);
    ++completed_count[thread_id];
  }
}

void StatisticsThread(std::string file_name) {
  std::ofstream ofs;
  ofs.open(file_name, std::ios::out);

  int finished = 0;
  int seconds = 0;
  while (finished < total_count - thread_num) {
    int new_finished = 0;
    for (int i = 0; i < thread_num; ++i) {
      new_finished += completed_count[i * 8];
    }

    int ops = (new_finished - finished) / 5;
    printf("[%d sec] %d operations; %d current ops/sec\n",
           seconds, new_finished, ops);

    ofs << seconds << " " << ops << std::endl;

    finished = new_finished;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    seconds += 5;
  }

  ofs.close();
}

typedef void(* TestFunction)(DB*, int);

void DoTest(DB* db, TestFunction test_func, std::string file_name) {
  memset(completed_count, 0, thread_num * 64);

  std::thread threads[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(test_func, db, i);
  }

  std::thread statistic_thread = std::thread(StatisticsThread, file_name);

  for (auto& thread : threads) {
    thread.join();
  }

  statistic_thread.join();
}

TestFunction Functions[3] = {Test1Thread, Test2Thread, Test3Thread};

int main(int argc, char* argv[]) {
  auto func = Functions[argc == 1 ? 2 : std::atoi(argv[1]) - 1];

  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  options.OptimizeLevelStyleCompaction();

  DB* db;
  DB::Open(options, "/tmp/db_test", &db);

  // Step 1: uniform
  DoTest(db, UniformThread, "/tmp/load_ops.txt");

  // Step2: zipfian
  final_samples.reserve(total_count);
  std::uniform_int_distribution<int> dist(0, sample_range - 1);
  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};

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
      value = freqs[value] == 1
                  ? dist(rng) :
                  (int)(fnvhash64(value) % 50000000) + left[value % 2];
    }
  }

  DoTest(db, Test3Thread, "/tmp/run_ops.txt");

  db->Close();

  delete db;

  return 0;
}