// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <random>
#include <vector>
#include <set>
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
  options.write_buffer_size = 4 * 1048576;
  options.level0_file_num_compaction_trigger = 4;
  options.use_direct_io_for_flush_and_compaction = true;
  options.use_direct_reads = true;


  std::vector<Slice> partition_keys(10, "");
  std::vector<std::string> values(10);
  for (int i = 1; i <= 9; i++) {
    values[i] = randomString(64);
    partition_keys[i] = Slice(values[i]);
  }
  std::sort(partition_keys.begin(), partition_keys.end());

  for (auto& s : partition_keys) {
    std::cout << "partition " << s.ToString(false) << std::endl;
  }

  options.partition_keys = partition_keys;
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
  s = db->Get(ReadOptions(), "key1", &value);

  for (int i = 0; i < 100000; i++) {
    db->Put(WriteOptions(), randomString(64), randomString(128));
  }

  s = db->Get(ReadOptions(), "key1", &value);
  assert(s.ok());
  assert(value == "value1");

  std::cout << " insert finished." << std::endl;

  for (int i = 0; i < 10000; i++) {
    // test random get
    db->Get(ReadOptions(), randomString(64), &value);
  }

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
