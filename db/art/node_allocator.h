//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <atomic>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include "macros.h"
#include "concurrent_queue.h"

namespace ROCKSDB_NAMESPACE {

// Forward declaration
struct NVMNode;

class NodeAllocator {
 public:
  NodeAllocator(const DBOptions& options, bool recovery);

  void Reset();

  NVMNode* GetHead();

  size_t GetNumFreePages() {
    return free_nodes_.size();
  }

  NVMNode* AllocateNode();

  void DeallocateNode(NVMNode* node);

  void FreeNodes();

  int64_t relative(NVMNode* node);

  NVMNode* absolute(int64_t offset);

 private:
  char* pmemptr_;

  int64_t total_size_;

  std::atomic<int> num_free_;

  TQueueConcurrent<char*> waiting_nodes_;

  TQueueConcurrent<char*> free_nodes_;
};

void InitializeNodeAllocator(const DBOptions& options, bool recovery = false);

NodeAllocator* GetNodeAllocator();

} // namespace ROCKSDB_NAMESPACE