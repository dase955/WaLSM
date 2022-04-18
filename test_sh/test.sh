#!/bin/bash

value_array=(32 64 128)
test_all_size=81920000000   #8G

pmem_path="/mnt/chen/test"

bench_benchmarks="fillrandom, stats, wait, clean_cache, stats, readrandom, stats, clean_cache"
bench_readnum="1000000"
max_background_jobs="3"

bench_file_path="$(dirname $PWD)/db_bench"
bench_file_dir="$(dirname $PWD)"
if [ ! -f "${bench_file_path}" ];then
  bench_file_path="$PWD/db_bench"
  bench_file_dir="$PWD"
  if [ ! -f "${bench_file_path}" ];then
    echo "Error:${bench_file_path} or $(dirname $PWD )/db_bench not find!"
    exit 1
  fi
fi

RUN_ONE_TEST() {
    const_params="
    --db=$pmem_path \
    --wal_dir=$pmem_path \
    --pmem_path=$pmem_path \
    --benchmarks=$bench_benchmarks \
    --value_size=$bench_value \
    --num=$bench_num \
    --reads=$bench_readnum \
    --max_background_jobs=$max_background_jobs \
    "
    cmd="$bench_file_path $const_params >>out.out 2>&1"
    echo $cmd > out.out
    echo $cmd
    eval $cmd
}

CLEAN_CACHE() {
    if [ -n "$bench_db_path" ];then
        rm -f $bench_db_path/*
    fi
    sleep 2
    sync
    echo 3 > /proc/sys/vm/drop_caches
    sleep 2
}

COPY_OUT_FILE(){
    mkdir $bench_file_dir/result > /dev/null 2>&1
    res_dir=$bench_file_dir/result/value-$bench_value
    mkdir $res_dir > /dev/null 2>&1
    \cp -f $bench_file_dir/compaction.csv $res_dir/
    \cp -f $bench_file_dir/OP_DATA $res_dir/
    \cp -f $bench_file_dir/OP_TIME.csv $res_dir/
    \cp -f $bench_file_dir/out.out $res_dir/
    \cp -f $bench_file_dir/Latency.csv $res_dir/
    \cp -f $bench_db_path/OPTIONS-* $res_dir/
}

RUN_ALL_TEST() {
    for value in "${value_array[@]}"; do
        CLEAN_CACHE
        bench_value="$value"
        bench_num="`expr $test_all_size / $bench_value`"

        RUN_ONE_TEST
        if [ $? -ne 0 ];then
            exit 1
        fi
        COPY_OUT_FILE
        sleep 5
    done
}

RUN_ALL_TEST