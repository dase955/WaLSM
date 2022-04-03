//
// Created by joechen on 2022/2/22.
//
// We use adaptive radix tree(ART) as our main data structure.
// Inner nodes(non-leaf nodes) are stored in dram, and leaf nodes are stored in NVM.
// Data in nodes are unsorted, fingerprints_ are used to speed up query efficiency.
// There have some assumption about input data:
// 1. 0th and 255th node will never be used
// 2. Key value size_ cannot exceed 64M
// TODO:
// 1. keys smaller than 8byte can be directly stored in node

#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <atomic>
#include <mutex>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

// Forward declarations
struct NVMNode;
struct HeatGroup;
struct ArtNodeHeader;
class VLogManager;
struct KVStruct;

struct InnerNode {
  uint64_t     buffer_[32];             // 256B buffer_
  uint8_t      hll_[64];        // 64B hyper log log

  // These values are rarely modified, put them in one cache-line

  // Pointer to heat group
  HeatGroup    *heat_group_;

  // Pointer to art node, backup is used for reallocate
  ArtNodeHeader *art, *artBackup;

  // Pointer to NVM node, backup is used for compaction
  NVMNode       *nvm_node_, *backup_nvm_node_;
  InnerNode     *last_child_node_;
  InnerNode     *next_node_;
  char          padding[8]{};

  Timestamps    ts;

  uint64_t      vptr_;

  // node status_, see macros.h
  uint32_t      status_;

  // estimated size_ of key and value in this node
  int32_t       estimated_size_;

  // total size_ in buffer_
  int32_t       buffer_size_;

  OptLock               opt_lock_;

  // Used for inserting new node into linked list
  SpinLock              link_lock_;

  // Used for flush and split operation
  std::mutex            flush_mutex_;

  InnerNode();
};

class GlobalMemtable {
 public:
  GlobalMemtable() : root_(nullptr), head_(nullptr), tail_(nullptr) {}

  void Put(Slice &slice, uint64_t vptr);

  bool Get(std::string &key, std::string &value);

  void InitFirstTwoLevel();

  void SetVLogManager(VLogManager *vlog_manager);

 private:
  bool FindKey(InnerNode *leaf, std::string &key, std::string &value);

  void SqueezeNode(InnerNode *leaf);

  void SplitNode(InnerNode *oldLeaf);

  // Optimize split below level 5, since we store first three prefixes in hash(key)
  void SplitLeafBelowLevel5(InnerNode *leaf);

  void InsertIntoLeaf(InnerNode *leaf, KVStruct &kvInfo, int level);

  InnerNode *root_;
  InnerNode *head_;
  InnerNode *tail_;

  VLogManager *vlog_manager_;
};

} // namespace ROCKSDB_NAMESPACE