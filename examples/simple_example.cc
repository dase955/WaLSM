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

int      thread_num = 32;
int      total_count = 320000000;
int      sample_range = 1000000000;
int      count_per_thread = total_count / thread_num;
int64_t* completed_count = new int64_t[thread_num * 8];
std::vector<int> final_samples;
std::atomic<int64_t> counter{0};

class ZipfianGenerator {
 public:
  ZipfianGenerator(double alpha) : alpha_(alpha) {
    InitProbs();
  }

  ~ZipfianGenerator() {
    delete[] sum_probs_;
  }

  long nextValue() {
    long res = 0;
    double z;
    do {
      z = rand_val(rng);
    } while ((z == 0) || (z == 1));

    int low = 1;
    int high = sample_range;
    int mid;
    do {
      mid = (low + high) / 2;
      if (sum_probs_[mid] >= z && sum_probs_[mid - 1] < z) {
        res = mid;
        break;
      } else if (sum_probs_[mid] >= z) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    } while (low <= high);

    return res;
  }

 private:
  void InitProbs() {
    printf("Start InitProbs\n");

    double c = 0.0;
    for (int i = 1; i <= sample_range; i++) {
      c = c + (1.0 / pow((double)i, alpha_));
    }
    c = 1.0 / c;

    sum_probs_ = new double[sample_range + 1];
    sum_probs_[0] = 0;
    for (int i = 1; i <= sample_range; ++i) {
      sum_probs_[i] = sum_probs_[i - 1] + c / pow((double)i, alpha_);
    }

    printf("Final value: %f\n", (float)sum_probs_[sample_range]);
  }

  double* sum_probs_;

  double alpha_;

  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};
  std::uniform_real_distribution<> rand_val{0.0, 1.0};
};

class ZipfianGeneratorYCSB {
 public:
  ZipfianGeneratorYCSB(long min, long max, double zipfianconstant_, double zetan_) {
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

inline uint64_t GetMicros() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec) * 1000000 + tv.tv_usec;
}

inline uint64_t GetTime() {
  static uint64_t start_time = GetMicros();
  return GetMicros() - start_time;
}

///////////////////////////////////////////////////////////////////////////////

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

void YCSBLoadThread(DB* db, int thread_id) {
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

void CustomWorkloadThread(DB* db, int thread_id) {
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

void ShuffleSamples(std::unordered_map<int, int>& freqs,
                    std::unordered_map<int, int>& modified) {
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

void CheckFreq() {
  std::unordered_map<int, int> freqs;
  for (auto& value : final_samples) {
    ++freqs[value];
  }

  int cold_count = 0;
  for (auto& pair : freqs) {
    cold_count += (pair.second == 1);
  }

  int hot_count = freqs.size() - cold_count;
  float hot_freq = (total_count - cold_count) / (float)total_count * 100.f;
  printf("Sample generated. hot count = %d(%.2f), cold count = %d\n",
         hot_count, hot_freq, cold_count);
}

void GenerateSamples() {
  final_samples.resize(total_count);

  ZipfianGenerator gen(0.98);
  std::unordered_map<int, int> freqs;

  for (int i = 0; i < total_count; ++i) {
    int value = gen.nextValue();
    final_samples[i] = value;
    freqs[value]++;
  }

  CheckFreq();

  std::unordered_map<int, int> modified;
  ShuffleSamples(freqs, modified);
  for (auto& value : final_samples) {
    value = modified[value];
  }

  CheckFreq();
}

void GenerateSamplesYCSB() {
  final_samples.resize(total_count);
  std::uniform_int_distribution<int> dist(0, 1000000000 - 1);
  std::random_device rd;
  std::default_random_engine rng = std::default_random_engine{rd()};

  {
    ZipfianGeneratorYCSB gen(0, 1000000000L, 0.99, 26.46902820178302);
    std::unordered_map<int, int> freqs;
    std::unordered_map<int, int> modified;
    int left[2] = {400000000, 800000000};
    int interval = 50000000;
    int mod = interval * 2;

    for (int i = 0; i < total_count; ++i) {
      long value = gen.nextValue();
      freqs[value]++;
      final_samples[i] = (int)value;
    }

    CheckFreq();

    for (auto& pair : freqs) {
      int old_value = pair.first;
      if (pair.second == 1) {
        modified[old_value] = dist(rng);
      } else {
        int new_val = fnvhash64(old_value) % mod;
        modified[old_value] = left[new_val / interval] + (new_val % interval);
      }
    }

    for (auto& value : final_samples) {
      value = modified[value];
    }

    CheckFreq();
  }
}

typedef void(*TestFunction)(DB*, int);

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

int main(int argc, char* argv[]) {
  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  options.OptimizeLevelStyleCompaction();

  std::string test_name = "art";
  /*if (argc >= 2) {
    test_name = argv[1];
  }
  if (argc >= 3) {
    total_count = atoi(argv[2]);
  }*/

  if (argc >= 2) {
    options.max_rewrite_count = atoi(argv[1]);
  }
  if (argc >= 3) {
    options.rewrite_threshold = atoi(argv[2]);
  }

  printf("max_rewrite_count=%d, rewrite_threshold=%d\n",
         options.max_rewrite_count, options.rewrite_threshold);

  std::string db_path = "/tmp/db_test_" + test_name;
  std::string ops_path =  "/tmp/run_ops_" + test_name;

  DB* db;
  DB::Open(options, db_path, &db);

  // Step 1: uniform
  // auto start_time1 = GetTime();
  // DoTest(db, UniformThread, "/tmp/load_ops.txt");
  // printf("Load phase: %.3f, %.3f\n", start_time1 * 1e-6, GetTime() * 1e-6);

  GenerateSamples();
  auto start_time2 = GetTime();
  DoTest(db, CustomWorkloadThread, ops_path);
  printf("Run phase:  %.3f, %.3f\n", start_time2 * 1e-6, GetTime() * 1e-6);

  db->Close();

  delete db;

  return 0;
}