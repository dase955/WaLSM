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
  NodeAllocator(bool recover = false);

  ~NodeAllocator();

  char* GetBase() {
    return pmemptr_;
  }

  size_t GetNumFreePages() {
    return free_pages_.size();
  }

  NVMNode* AllocateNode();

  void DeallocateNode(NVMNode* node);

  void recoverOnRestart();

  int64_t relative(NVMNode* node);

  NVMNode* absolute(int64_t offset);

 private:
  char* pmemptr_;

  std::atomic<int> num_free_{1048576};

  int64_t total_size_;

  TQueueConcurrent<char*> free_pages_;
};

NodeAllocator& GetNodeAllocator();

} // namespace ROCKSDB_NAMESPACE