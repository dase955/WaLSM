#!/bin/bash

DB_PATH=/tmp/db_nvm_l0
RES_PATH=/home/crh/result_crh
NVM_PATH=/mnt/chen

Clean() {
    rm -rf $DB_PATH
    rm -rf $NVM_PATH/rocksdb_l0
    rm -rf $NVM_PATH/rocksdb_log
}

Run() {
  echo $1
  Clean
  ./ycsb -load -run -db rocksdb -P workloads/workloada_$1 -P rocksdb/rocksdb.properties -s > $1.log 2>&1
  mv $DB_PATH/compaction.csv $RES_PATH/nvml0_098_$1.txt
}

Run 0
Run 50
Run 100
Run 25
Run 75