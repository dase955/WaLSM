#!/bin/bash

RunWorkload() {
  awk -F "=" '{print $2} NR==3{exit}' rocksdb/rocksdb.properties | xargs rm -rf
  ./ycsb -load -run -db rocksdb -P workloads/workload"$1" -P rocksdb/rocksdb.properties -s
}

RunWorkload b