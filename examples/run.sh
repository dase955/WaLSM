make simple_example -j8
sudo rm -f /tmp/compaction_art.csv
sudo rm -f /mnt/chen/nodememory
sudo rm -rf /tmp/db_test_art
sudo rm -f /tmp/run_ops_art
sudo rm -f /tmp/debug_art.txt
sudo taskset -c 0-17,36-53 ./simple_example