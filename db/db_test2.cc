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

int      thread_num = 8;
int      total_count = 320000000;
int      sample_range = 1000000000;
int      count_per_thread = total_count / thread_num;
std::vector<int> final_samples;
std::atomic<int> counter{0};

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
  std::cout << std::endl;
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

void GenerateSamples() {
  final_samples.resize(total_count);

  ZipfianGenerator gen(0.98);
  std::unordered_map<int, int> freqs;

  for (int i = 0; i < total_count; ++i) {
    int value = gen.nextValue();
    final_samples[i] = value;
    freqs[value]++;
  }

  std::unordered_map<int, int> freq_freq;
  for (auto& pair : freqs) {
    freq_freq[pair.second]++;
  }
  std::vector<std::pair<int, int>> freq_freq_sorted;
  freq_freq_sorted.reserve(freq_freq.size());
  for (auto& pair : freq_freq) {
    freq_freq_sorted.emplace_back(pair);
  }
  std::sort(freq_freq_sorted.begin(), freq_freq_sorted.end(),
            [](std::pair<int, int>& p1, std::pair<int, int>& p2){
              return p1.first < p2.first;
            });
  std::ofstream ofs("/tmp/freq.txt");
  int sum = 0;
  for (auto& pair : freq_freq_sorted) {
    sum += (pair.first * pair.second);
    ofs << pair.first << ", " << pair.second << ", " << ((float)sum * 1053.0 / 1024.0 / 1024.0 / 1024.0) << "G" << std::endl;
  }
  ofs.close();

  CheckFreq();

  std::unordered_map<int, int> modified;
  ShuffleSamples(freqs, modified);
  for (auto& value : final_samples) {
    value = modified[value];
  }

  CheckFreq();
}

void LoadOrWriteSamples() {
  int64_t sample_file_size = 4ll << 30;
  std::string sample_file = "/tmp/sample_data";

  int fd;
  struct stat buffer;
  bool file_exist = stat(sample_file.c_str(), &buffer) == 0;
  if (!file_exist) {
    std::cout << "Create file and generate samples" << std::endl;
    fd = open(sample_file.c_str(), O_RDWR|O_CREAT, 00777);
    assert(-1 != fd);
    lseek(fd, sample_file_size - 1, SEEK_SET);
    write(fd, "", 1);
  } else {
    std::cout << "Read samples from files" << std::endl;
    fd = open(sample_file.c_str(), O_RDWR, 00777);
  }

  char* ptr = (char*)mmap(
      nullptr, sample_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);

  final_samples.resize(total_count);
  if (file_exist) {
    memcpy(final_samples.data(), ptr, total_count * sizeof(int));
    CheckFreq();
  } else {
    GenerateSamples();
    memcpy(ptr, final_samples.data(), total_count * sizeof(int));
  }

  munmap(ptr, sample_file_size);
}

TEST_F(DBTest3, MockEnvTest) {
  LoadOrWriteSamples();

  DB* db;

  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  options.OptimizeLevelStyleCompaction();

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
