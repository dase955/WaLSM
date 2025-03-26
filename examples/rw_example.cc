#include <fcntl.h>
#include <algorithm>
#include <set>
#include <thread>
#include <unordered_set>
#include <utility>
#include <unistd.h>
#include <iostream>
#include <random>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

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
  return "user" + std::string(9 - s.length(), '0') + s;
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
      std::cout << key << " Error!" << std::endl;
    }
  }
}

void StatisticsThread() {
  int prev_completed = 0;
  int seconds = 0;
  while (prev_completed < total_count) {
    int new_completed = counter.load(std::memory_order_relaxed);
    int ops = (new_completed - prev_completed) / 5;
    printf("[%d sec] %d operations; %d current ops/sec\n",
           seconds, new_completed, ops);

    prev_completed = new_completed;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    seconds += 5;
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

int main() {
  Options options;
  options.create_if_missing = true;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;
  options.enable_pipelined_write = true;
  options.nvm_path = "/pg_wal/ycc/memory_art";
  options.compression = rocksdb::kNoCompression;

  DB* db;
  DB::Open(options, "/tmp/tmp_data/db_test_art", &db);

  std::thread read_threads[thread_num];
  std::thread write_threads[thread_num];
  std::thread check_threads[thread_num];
  for (int i = 0; i < thread_num; ++i) {
    write_threads[i] = std::thread(ZipfianPutThread, db);
    //read_threads[i] = std::thread(GetThread, db);
  }

  auto st = std::thread(StatisticsThread);

  for (auto & thread : read_threads) {
    //thread.join();
  }

  for (auto & thread : write_threads) {
    thread.join();
  }

  st.join();

  std::this_thread::sleep_for(std::chrono::seconds(10));

  std::cout << "Start test get" << std::endl;

  for (int i = 0; i < thread_num; ++i) {
    check_threads[i] = std::thread(CheckThread, db);
  }

  for (int i = 0; i < thread_num; ++i) {
    check_threads[i].join();
  }

  delete db;
}
