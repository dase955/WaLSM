//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

struct NVMNodeMeta {
  uint64_t header = 0;
  int64_t next1 = -1;  // We only store relative offset in meta
  int64_t next2 = -1;
  void*   dram_pointer_ = nullptr;
  uint8_t fingerprints_[224] = {0};
};

struct NVMNode {
  NVMNodeMeta meta;
  uint64_t compactionData[32] = {0};
  uint64_t data[448] = {0};
};

} // namespace ROCKSDB_NAMESPACE