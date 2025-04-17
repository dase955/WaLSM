cd ..
# make clean
make static_lib -j32

cd YCSB
make clean && make DEBUG_BUILD=1

