cd ..
rm -rf ../log/*
make clean
make static_lib -j32

# mv librocksdb_debug.a librocksdb.a

cd YCSB
make clean && make -j4

rm -rf /mnt/nvme0n1/guoteng/walsmtest/tmp/db_nvm_l0
rm -rf /mnt/pmem0.7/guoteng/nodememory
