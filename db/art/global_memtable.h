//
// Created by joechen on 2022/2/22.
//
// We use adaptive radix tree(ART) as our main data structure.
// Inner nodes(non-leaf nodes) are stored in dram, and leaf nodes are stored in NVM.
// Data in nodes are unsorted, fingerprints_ are used to speed up query efficiency.
// There have some assumption about input data:
// 1. Key value size_ cannot exceed 64M
// TODO:
// 1. keys smaller than 8byte can be directly stored in node

#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <util/mutexlock.h>
#include <util/autovector.h>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

// Forward declarations
struct NVMNode;
struct HeatGroup;
struct ArtNode;
class HeatGroupManager;
class VLogManager;
struct KVStruct;

struct InnerNode {
  uint64_t   buffer_[32];     // 256B buffer
  uint8_t    hll_[64];        // hyper log log use 64 buckets

  HeatGroup* heat_group_;
  ArtNode*   art;

  // Backup is used for compaction
  NVMNode*   nvm_node_;
  NVMNode*   backup_nvm_node_;

  // Used for inserting new node
  InnerNode* last_child_node_;
  InnerNode* parent_node_;

  InnerNode* next_node_;
  uint64_t   vptr_;

  std::mutex flush_mutex_;      // Used for flush and split operation
  SpinMutex  link_lock_;
  OptLock    opt_lock_;
  int32_t    estimated_size_;   // Estimated kv size in this node
  uint32_t   status_;           // node status, see macros.h
  int64_t    oldest_key_time_;  // Just for compatibility

  std::shared_mutex shared_mutex;

  InnerNode();
};

class GlobalMemtable {
 public:
  void InitFirstLevel();

  GlobalMemtable()
      : root_(nullptr), vlog_manager_(nullptr),
        group_manager_(nullptr), env_(nullptr) {}

  GlobalMemtable(VLogManager* vlog_manager,
                 HeatGroupManager* group_manager,
                 Env* env, bool recovery = false);

  ~GlobalMemtable();

  void Recovery();

  InnerNode* RecoverNonLeaf(InnerNode* parent, int level, HeatGroup*& group);

  void Put(Slice& slice, uint64_t base_vptr, size_t count);

  bool Get(std::string& key, std::string& value, Status* s);

  InnerNode* FindInnerNodeByKey(Slice& key, size_t& level,
                                bool& stored_in_nvm);

 private:
  void Put(Slice& key, KVStruct& kv_info);

  bool FindKeyInInnerNode(InnerNode* leaf, size_t level,
                          std::string& key, std::string& value, Status* s);

  void InsertIntoLeaf(InnerNode* leaf, KVStruct& kv_info, size_t level);

  void SqueezeNode(InnerNode* leaf);

  // Split leaf node and store node that still need split
  void SplitLeaf(InnerNode* leaf, size_t level,
                 InnerNode** node_need_split);

  int32_t ReadFromNVM(NVMNode* nvm_node, size_t level,
                   uint64_t& leaf_vptr, autovector<KVStruct>* data);

  int32_t ReadFromVLog(NVMNode* nvm_node, size_t level,
                    uint64_t& leaf_vptr, autovector<KVStruct>* data);

  InnerNode* root_;

  // TODO: Use prefixes to fast search inner nodes
  std::unordered_map<std::string, InnerNode*> prefixes_;

  VLogManager* vlog_manager_;

  HeatGroupManager* group_manager_;

  Env* env_;
};

} // namespace ROCKSDB_NAMESPACE