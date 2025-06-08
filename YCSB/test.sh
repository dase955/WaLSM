cd ..
rm -rf log/*
make clean
make static_lib -j32

#mv librocksdb_debug.a librocksdb.a

cd YCSB
make clean && make -j4

rm -rf /mnt/nvme0n1/guoteng/walsmtest/tmp/gt_test
rm -rf /mnt/pmem0.8/guoteng/nodememory

#gdb --args ./ycsb -load -run -db rocksdb -P workloads/workloadt -P rocksdb/rocksdb.properties -p threadcount=8 -s
./ycsb -load -run -db rocksdb -P workloads/workloadt -P rocksdb/rocksdb.properties -p threadcount=8 -p sleepafterload=60 -s
