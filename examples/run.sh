#!/bin/bash
make simple_example -j8
sudo rm -f /tmp/compaction_art.csv
sudo rm -rf /mnt/chen/*
sudo rm -rf /tmp/db_test_nvm_l0
sudo rm -f /tmp/run_ops_nvm_l0
sudo rm -f /tmp/debug_art.txt
sudo taskset -c 0-17 ./simple_example.cc
