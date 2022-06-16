// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <random>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

using namespace ROCKSDB_NAMESPACE;

std::string kDBPath = "/Users/chenlixiang/rocksdb_test";

std::string randomString(int len) {
  char tmp;
  std::string buf;

  std::random_device dev;
  std::default_random_engine random(dev());

  for (int i = 0; i < len; i++) {
    tmp = random() % 36;
    tmp += (tmp < 10) ? '0' : ('a'-10);
    buf += tmp;
  }

  return buf;
}

int main() {
  DB* db;
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeUniversalStyleCompaction();
  options.use_direct_io_for_flush_and_compaction = true;
  // create the DB if it's not already present
  options.create_if_missing = true;
//  options.target_file_size_base = 4 * 1048576;
  options.write_buffer_size = 4 << 20;
  options.level0_file_num_compaction_trigger = 4;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;



  // open DB
  Status s = DB::Open(options, kDBPath, &db);
  assert(s.ok());

  std::cout << " db initialized " << std::endl;

  db->Put(WriteOptions(), "key1", "value1");

  // Put key-value
//  s = db->Put(WriteOptions(), "key1", "value");
//  assert(s.ok());
  std::string value;
  // get value
//  s = db->Get(ReadOptions(), "key1", &value);

  std::unordered_map<std::string, std::string> kvs;
  std::unordered_set<std::string> repeat_keys;
  for (int i = 0; i < 10000000; i++) {
    std::string my_key = randomString(128);
    std::string my_value = randomString(200);
    if (kvs.find(my_key) != kvs.end()) {
      repeat_keys.insert(my_key);
      kvs[my_key] = my_value;
    } else if (i % 30 == 1) {
      kvs[my_key] = my_value;
    }
    db->Put(WriteOptions(), my_key, my_value);
  }

  s = db->Get(ReadOptions(), "key1", &value);
  assert(s.ok());
  assert(value == "value1");

  std::cout << " insert finished." << std::endl;

  for (int i = 0; i < 100; i++) {
    // test random get
    db->Get(ReadOptions(), randomString(64), &value);
  }

  // check data
  int no_hit = 0, repeat = 0, bad_value = 0;
  for (auto& kv : kvs) {
    s = db->Get(ReadOptions(), kv.first, &value);
    if (s.IsNotFound()) {
      no_hit++;
    }
    else if (value != kv.second) {
      if (repeat_keys.find(kv.first) != repeat_keys.end()) {
        std::cout << "REPEAT KEY ERROR!" << "$";
        repeat++;
      }
      std::cout << "key: " << kv.first << " expect: " << kv.second
      << " got: " << value << std::endl;
      bad_value++;
    }
  }
  printf("%d/%d not_found. %d/%d repeat/bad values. total repeat data size %d", no_hit, (int) kvs.size(),
         repeat, bad_value, (int)repeat_keys.size());

  // atomically apply a set of updates
  {
    WriteBatch batch;
    batch.Delete("key1");
    batch.Put("key2", "value");
    s = db->Write(WriteOptions(), &batch);
  }

  s = db->Get(ReadOptions(), "key1", &value);
  assert(s.IsNotFound());

  db->Get(ReadOptions(), "key2", &value);
  assert(value == "value");

  {
    PinnableSlice pinnable_val;
    db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
    assert(pinnable_val == "value");
  }

  {
    std::string string_val;
    // If it cannot pin the value, it copies the value to its internal buffer.
    // The intenral buffer could be set during construction.
    PinnableSlice pinnable_val(&string_val);
    db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
    assert(pinnable_val == "value");
    // If the value is not pinned, the internal buffer must have the value.
    assert(pinnable_val.IsPinned() || string_val == "value");
  }

  PinnableSlice pinnable_val;
  s = db->Get(ReadOptions(), db->DefaultColumnFamily(), "key1", &pinnable_val);
  assert(s.IsNotFound());
  // Reset PinnableSlice after each use and before each reuse
  pinnable_val.Reset();
  db->Get(ReadOptions(), db->DefaultColumnFamily(), "key2", &pinnable_val);
  assert(pinnable_val == "value");
  pinnable_val.Reset();
  // The Slice pointed by pinnable_val is not valid after this point

  delete db;

  return 0;
}
