//
// Created by joechen on 2022/2/22.
//
// We use adaptive radix tree(ART) as our main data structure.
// Inner nodes(non-leaf nodes) are stored in dram, and leaf nodes are stored in NVM.
// Data in nodes are unsorted, fingerprints are used to speed up query efficiency.
// There have some assumption about input data:
// 1. Key value size cannot exceed 64M
// TODO:
// 1. keys smaller than 8byte can be directly stored in node
// 2. store vptr in support node instead of inner node

#pragma once
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include "table/internal_iterator.h"
#include "util/autovector.h"
#include "util/mutexlock.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "rwspinlock.h"
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

// Forward declarations
struct NVMNode;
struct HeatGroup;
struct ArtNode;
class HeatGroupManager;
class VLogManager;
struct KVStruct;
class GlobalMemTableIterator;

struct InnerNode {
  uint64_t    buffer_[32];     // 256B buffer
  uint8_t     hll_[64];        // hyper log log use 64 buckets

  HeatGroup*  heat_group_;

  // backup art pointer is used for compaction
  ArtNode*    art;
  ArtNode*    backup_art;

  // Backup nvm node is used for compaction
  NVMNode*    nvm_node_;
  NVMNode*    backup_nvm_node_;

  // support node is used to simplify insert operation,
  // it itself doesn't store any data.
  InnerNode*  support_node;
  InnerNode*  parent_node;
  InnerNode*  next_node;

  uint64_t    vptr_;
  uint64_t    hash_;

  SharedMutex share_mutex_; // Used for flush and split operation
  RWSpinLock  art_rw_lock_; // Protect art pointer
  RWSpinLock  link_lock_;   // Protect last child node
  RWSpinLock  vptr_lock_;

  OptLock     opt_lock_;
  int32_t     estimated_size_;   // Estimated kv size in this node
  int32_t     squeezed_size_;
  uint32_t    status_;           // node status, see macros.h
  int64_t     oldest_key_time_;  // Just for compatibility

  InnerNode();
};

class GlobalMemtable {
  friend class GlobalMemTableIterator;
 public:
  void InitFirstLevel();

  GlobalMemtable()
      : root_(nullptr), vlog_manager_(nullptr),
        group_manager_(nullptr), env_(nullptr) {}

  GlobalMemtable(VLogManager* vlog_manager,
                 HeatGroupManager* group_manager,
                 Env* env, bool recovery = false);

  ~GlobalMemtable();

  void Reset();

  void Recovery();

  InternalIterator* NewIterator(const ReadOptions& read_options);

  InnerNode* RecoverNonLeaf(InnerNode* parent, int level, HeatGroup*& group);

  void Put(Slice& slice, uint64_t base_vptr, size_t count);

  bool Get(std::string& key, std::string& value, Status* s);

  InnerNode* FindInnerNodeByKey(const Slice& key, size_t& level,
                                bool& stored_in_nvm);

 private:
  friend class Compactor;

  void PutRecover(uint64_t vptr);

  void Put(Slice& key, KVStruct& kv_info, bool update_heat = true);

  bool FindKeyInInnerNode(InnerNode* leaf, size_t level,
                          std::string& key, std::string& value, Status* s);

  bool ReadInNVMNode(NVMNode* nvm_node, uint64_t hash,
                     std::string& key, std::string& value, Status* s);

  void InsertIntoLeaf(InnerNode* leaf, KVStruct& kv_info,
                      size_t level, bool update_heat = true);

  bool SqueezeNode(InnerNode* leaf);

  // Split leaf node and store node that still need split
  void SplitLeaf(InnerNode* leaf, size_t level,
                 InnerNode** node_need_split);

  int32_t ReadFromNVM(NVMNode* nvm_node, size_t level,
                      uint64_t& leaf_vptr, uint64_t& leaf_hash,
                      autovector<KVStruct>* data);

  int32_t ReadFromVLog(NVMNode* nvm_node, size_t level,
                       uint64_t& leaf_vptr, uint64_t& leaf_hash,
                       autovector<KVStruct>* data);

  InnerNode* root_;

  InnerNode* tail_;

  // TODO: Use prefixes to fast search inner nodes
  std::unordered_map<std::string, InnerNode*> prefixes_;

  VLogManager* vlog_manager_;

  HeatGroupManager* group_manager_;

  Env* env_;
};

} // namespace ROCKSDB_NAMESPACE