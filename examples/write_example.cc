// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <random>
#include <algorithm>
#include <thread>
#include <cassert>
#include <iostream>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

using namespace ROCKSDB_NAMESPACE;

std::string kDBPath = "/tmp/rocksdb_simple_example";

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

std::atomic<int64_t> counter{0};

std::string next_key() {
  int c = counter++;
  auto keynum = fnvhash64(c);
  std::string key = std::to_string(keynum);
  if (c % 500000 == 0) {
    printf("Write %d records.\n", c);
  }
  return "user" + key;
}

std::string next_key(int64_t c) {
  auto keynum = fnvhash64(c);
  std::string key = std::to_string(keynum);
  return "user" + key;
}

void MultiThreadTest(DB *db, int thread_id) {
  for (int i = 0; i < count_per_thread; ++i) {
    std::string key = next_key();
    std::string value = repeat(key);
    assert(db->Put(WriteOptions(), key, value).ok());
  }
}

int main() {
  Options options;
  options.create_if_missing = true;
  options.enable_pipelined_write = true;

  DB* db;

  assert(DB::Open(options, "/tmp/db_test", &db).ok());

  int n = 0;

  std::thread threads[thread_num];
  for (auto& thread : threads) {
    thread = std::thread(MultiThreadTest, db, n++);
  }

  for (auto & thread : threads) {
    thread.join();
  }

  std::cout << "Start test get" << std::endl;
  for (int i = 0; i < total_count; i += 16) {
    std::string key = next_key(i);
    std::string res;
    std::string expected = repeat(key);
    auto status = db->Get(ReadOptions(), key, &res);
    assert(!status.ok() || res == expected);
    if (!status.ok()) {
      std::cout << key << " Error!" << std::endl;
    }
  }
  std::cout << "Test get done" << std::endl;

  db->Close();

  delete db;
}
