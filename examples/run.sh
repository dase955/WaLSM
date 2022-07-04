#!/bin/bash

Cleanup() {
    rm -f /home/crh/compaction_art.csv
    rm -f /home/crh/debug_art.txt
    rm -f /home/crh/run_ops_art
    rm -rf /home/crh/db_test_art
    rm -f /mnt/pmem1/crh/nodememory
}

#Cleanup
#./zipf_example 1.0 > 1.txt

#Cleanup
#./zipf_example 0.75 > 075.txt

Cleanup
./zipf_example 0.5 > 05.txt

#Cleanup
#./zipf_example 0.25 > 025.txt

#Cleanup
#./zipf_example 0.0 > 0.txt
