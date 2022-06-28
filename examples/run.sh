sudo rm -rf /tmp/db_test_art
sudo rm -f /mnt/chen/nodememory
sudo rm -f /tmp/run_ops_art
sudo rm -f /tmp/debug_art.txt
sudo rm -f /tmp/compaction_art.csv
sudo taskset -c 0-17,36-53 ./simple_example
