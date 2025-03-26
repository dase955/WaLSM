## Introduction

Author: Guo Teng. Email: PRCguoteng@gmail.com

**Deprecated.** Output read workload file when run YCSB and divide workload into key ranges.

**Now we try to sample the Put key and generate key ranges, so dont use this tool any more**

## Build and run

simple run 

``` shell
cd workloads
g++ generator.cc -o generator
./generator
```

## Argument

there are three arguments need to be defined, just see ```generator.cc```

``` c++
// simple args, avoid using command line arg or config file
const std::string input_workload = "workload";
const std::string output_seperators = "seperators";
const int key_range_num = 1000;
```

## Workload

we use YCSB to generator workload file, need enable micro USE_WORKLOAD when compiling. 

modify this code in ```YCSB/rocksdb/rocksdb_db.cc RocksdbDB::ReadSingle(...)``` func

``` c++
DB::Status RocksdbDB::ReadSingle(const std::string &table, const std::string &key,
                                 const std::vector<std::string> *fields,
                                 std::vector<Field> &result) {
  std::string data;
  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), key, &data);
  #ifdef GEN_WORKLOAD
  std::fstream f;
	f.open("../workload/workload", std::ios::out | std::ios::app);
	f << key <<std::endl;
	f.close();
  #endif
  if (s.IsNotFound()) {
    return kNotFound;
  } else if (!s.ok()) {
    throw utils::Exception(std::string("RocksDB Get: ") + s.ToString());
  }
  if (fields != nullptr) {
    DeserializeRowFilter(result, data, *fields);
  } else {
    DeserializeRow(result, data);
    assert(result.size() == static_cast<size_t>(fieldcount_));
  }
  return kOK;
}
```

then run one workload under YCSB/workload only use **one thread**

After that, you will get the new workload file in workload dir

## Use Workload

Firstly, you should add -DWALSM_PLUS when compiling WaLSM code to enable heat_buckets

Then, you can see related macros in ```db/art/macros.h```

``` c++
// micros for HeatBuckets
#define SEPERATORS_PATH "/home/ycc/WaLSM/workload/seperators"
#define BUCKETS_PERIOD 20000
#define BUCKETS_ALPHA 0.2
```

modify ```SEPERATORS_PATH``` to your own generated seperators file path