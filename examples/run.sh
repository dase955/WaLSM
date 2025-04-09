#!/bin/bash

SingleTest() {
    sudo rm -rf /mnt/pmem0.7/guoteng/*
    sudo rm -rf /mnt/nvme0n1/guoteng/walsmtest/tmp/*
    numactl -N 1 ./ycsb $1 $2
    #mv /tmp/db_old_custom/compaction_art.txt /home/chen/result/art_reset_$1_$2.txt
}

#SingleTest 0.98 0.5
#SingleTest 0.98 0.25
#SingleTest 0.98 0.75
SingleTest 0.98 0
#SingleTest 0.98 1
