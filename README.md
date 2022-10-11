# WaLSM

WaLSM is a workload-aware KV store for NVM-SSD hybrid storage. It is designed for reducing write amplifications on SSD with NVM, and auto index tuning on varying workloads.

## Build

Our code is on `dev` branch.

Currently, we only support building with Makefile. Use

```
make -j32 static_lib
```

to build the static library.

## Test

We use the following settings for testing performance:

```C++
  opt.create_if_missing = true;
  opt.use_direct_io_for_flush_and_compaction = true;
  opt.use_direct_reads = true;
  opt.compression = rocksdb::kNoCompression;
  opt.compaction_style = rocksdb::kCompactionStyleUniversal;
  opt.IncreaseParallelism(32);
  opt.statistics = rocksdb::CreateDBStatistics();
  opt.nvm_path = nvm_path; // use nvm path here

  rocksdb::BlockBasedTableOptions block_based_options;
  block_based_options.pin_top_level_index_and_filter = false;
  block_based_options.pin_l0_filter_and_index_blocks_in_cache = false;
  block_based_options.cache_index_and_filter_blocks_with_high_priority = false;
  block_based_options.index_type = rocksdb::BlockBasedTableOptions::kTwoLevelIndexSearch;
  block_based_options.partition_filters = true;
  block_based_options.cache_index_and_filter_blocks = true;
  block_based_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  block_based_options.block_cache =
      rocksdb::NewLRUCache(static_cast<size_t>(128 * 1024 * 1024));
  opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(block_based_options));
  opt.memtable_prefix_bloom_size_ratio = 0.02;
```

Note that our compacting algorithm is based on `Universal Compaction`, using other RocksDB default compacting algorithm may cause unexpected behaviors.
