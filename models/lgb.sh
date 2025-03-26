rm lgb lgb.txt log.txt
gcc lgb.cc -o lgb -l_lightgbm -std=c++11 -lstdc++ -lpython3.12 -I/home/ycc/miniconda3/include/python3.12 -L/home/ycc/miniconda3/lib
./lgb > log.txt