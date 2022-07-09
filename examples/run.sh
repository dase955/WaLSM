#!/bin/bash

NVM_PATH=/mnt/chen
DB_PATH=/tmp/tmp_data/db_test_nvm_l0
OUT_PATH=/tmp/tmp_data/result

rm -rf ${NVM_PATH}/rocksdb_log/
rm -rf ${NVM_PATH}/rocksdb_l0/
rm -rf ${DB_PATH}

./mini_benchmark $1 > ops.txt
mv ops.txt ${OUT_PATH}/nvm_l0_$1.txt
mv ${DB_PATH}/compaction_nvm_l0.csv ${OUT_PATH}/compaction_nvm_l0_$1.csv