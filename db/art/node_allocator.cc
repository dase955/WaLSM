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

void InitializeNodeAllocator(const DBOptions& options, bool recovery) {
  Allocator = new NodeAllocator(options, recovery);
}

NodeAllocator* GetNodeAllocator() {
  return Allocator;
}

NVMNode* NodeAllocator::GetHead() {
  return (NVMNode*)pmemptr_;
}

NodeAllocator::NodeAllocator(const DBOptions& options, bool recovery)
    : total_size_(options.node_memory_size),
      num_free_(total_size_ / (int64_t)PAGE_SIZE){

  pmemptr_ = GetMappedAddress("nodememory");

  int total_num_page = num_free_.load();
  int* non_free_pages = new int[total_num_page];
  memset(non_free_pages, 0, sizeof(int) * total_num_page);

  if (recovery) {
    auto cur_node = (NVMNode*)pmemptr_;
    non_free_pages[0] = 1;
    while (true) {
      int64_t next_offset =
          GET_TAG(cur_node->meta.header, ALT_FIRST_TAG)
              ? cur_node->meta.next1 : cur_node->meta.next2;
      cur_node = absolute(next_offset);
      if (!cur_node) {
        break;
      }

      non_free_pages[next_offset / PAGE_SIZE] = 1;
    }
  }

  char* cur_ptr = pmemptr_;
  for (int i = 0; i < num_free_; ++i) {
    if (!non_free_pages[i]) {
      free_nodes_.emplace_back(cur_ptr);
    }
    cur_ptr += PAGE_SIZE;
  }

  delete[] non_free_pages;
}

NVMNode* NodeAllocator::AllocateNode() {
  char* ptr = free_nodes_.pop_front();
  memset(ptr, 0, 4096);
  auto nvm_node = (NVMNode*)ptr;
  return nvm_node;
}

void NodeAllocator::FreeNodes() {
  auto size = waiting_nodes_.size();
  while (size--) {
    free_nodes_.emplace_back(waiting_nodes_.pop_front());
  }
}

void NodeAllocator::DeallocateNode(NVMNode* node) {
  waiting_nodes_.emplace_back((char*)node);
}

int64_t NodeAllocator::relative(NVMNode* node) {
  return (char*)node - pmemptr_;
}

NVMNode* NodeAllocator::absolute(int64_t offset) {
  return offset == -1 ? nullptr : (NVMNode*)(pmemptr_ + offset);
}

} // namespace ROCKSDB_NAMESPACE