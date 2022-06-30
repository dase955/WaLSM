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
  // For leaf node, store estimated size of data,
  // and for non-leaf node, store vptr
  uint64_t node_info = 0;
  uint8_t fingerprints_[224] = {0};
};

struct NVMNode {
  NVMNodeMeta meta;
  // temp buffer is used when doing compaction or closing db,
  // to store data in dram
  uint64_t temp_buffer[32] = {0};
  uint64_t data[448] = {0};
};

} // namespace ROCKSDB_NAMESPACE