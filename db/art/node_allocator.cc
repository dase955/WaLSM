//
// Created by joechen on 2022/4/3.
//

#include "node_allocator.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cassert>
#include <unistd.h>

#include "nvm_manager.h"
#include "nvm_node.h"
#include "utils.h"

namespace ROCKSDB_NAMESPACE {

NodeAllocator* Allocator;

void InitializeNodeAllocator(const DBOptions& options) {
  Allocator = new NodeAllocator(options);
}

NodeAllocator* GetNodeAllocator() {
  return Allocator;
}

NodeAllocator::NodeAllocator(const DBOptions& options)
    : total_size_(options.node_memory_size),
      num_free_(total_size_ / (int64_t)PAGE_SIZE){

  pmemptr_ = GetMappedAddress("nodememory");

  char* cur_ptr = pmemptr_;
  for (int i = 0; i < num_free_; ++i) {
    free_pages_.emplace_back(cur_ptr);
    cur_ptr += PAGE_SIZE;
  }
}

NVMNode* NodeAllocator::AllocateNode() {
  char* ptr = free_pages_.pop_front();
  memset(ptr, 0, 512);
  auto nvm_node = (NVMNode*)ptr;
  return nvm_node;
}

void NodeAllocator::DeallocateNode(NVMNode* node) {
  free_pages_.emplace_back((char*)node);
}

int64_t NodeAllocator::relative(NVMNode* node) {
  return (char*)node - pmemptr_;
}

NVMNode* NodeAllocator::absolute(int64_t offset) {
  return offset == -1 ? nullptr : (NVMNode*)(pmemptr_ + offset);
}

} // namespace ROCKSDB_NAMESPACE