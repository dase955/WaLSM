#!/bin/bash

SingleTest() {
    sudo rm -rf /mnt/chen/*
    sudo rm -rf /tmp/db_old_custom
    taskset -c 1-32 ./custom $1
    mv /tmp/db_old_custom/compaction_art.txt /home/chen/result/matrixkv_custom_$1_modified.txt

    sudo rm -rf /mnt/chen/*
    sudo rm -rf /tmp/db_old_custom
    taskset -c 1-32 ./ycsb $1 1000000000
    mv /tmp/db_old_custom/compaction_art.txt /home/chen/result/matrixkv_ycsb_$1_10b_modified.txt

    sudo rm -rf /mnt/chen/*
    sudo rm -rf /tmp/db_old_custom
    taskset -c 1-32 ./ycsb $1 10000000000
    mv /tmp/db_old_custom/compaction_art.txt /home/chen/result/matrixkv_ycsb_$1_100b_modified.txt
}

SingleTest 0.9
SingleTest 0.95
SingleTest 0.98
SingleTest 1.05
SingleTest 1.10
