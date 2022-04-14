//
// Created by joechen on 2022/4/3.
//

#include "node_allocator.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <cassert>
#include <csignal>

#include "nvm_node.h"

namespace ROCKSDB_NAMESPACE {

NodeAllocator& GetNodeAllocator() {
  static NodeAllocator manager;
  return manager;
}

NodeAllocator::NodeAllocator(bool recover) {
  total_size_ = num_free_ * (int64_t)PAGE_SIZE;

  int fd = open("/home/joechen/NodeMemory", O_RDWR|O_CREAT, 00777);
  assert(-1 != fd);
  lseek(fd, total_size_ - 1, SEEK_SET);
  write(fd, "", 1);
  pmemptr_ = (char*)mmap(nullptr, total_size_, PROT_READ | PROT_WRITE,MAP_SHARED, fd, 0);
  close(fd);

  if (recover) {
    recoverOnRestart();
    return;
  }

  char* cur_ptr = pmemptr_;
  for (int i = 0; i < num_free_; ++i) {
    free_pages_.emplace_back(cur_ptr);
    cur_ptr += PAGE_SIZE;
  }
}

NodeAllocator::~NodeAllocator() {
  munmap(pmemptr_, total_size_);
}

NVMNode* NodeAllocator::AllocateNode() {
  char* ptr = free_pages_.pop_front();
  memset(ptr, 0, PAGE_SIZE);
  return (NVMNode*)ptr;
}

void NodeAllocator::DeallocateNode(NVMNode* node) {
  free_pages_.emplace_back((char*)node);
}

void NodeAllocator::recoverOnRestart() {

}

int64_t NodeAllocator::relative(NVMNode* node) {
  return (char*)node - pmemptr_;
}

NVMNode* NodeAllocator::absolute(int64_t offset) {
  return offset == -1 ? nullptr : (NVMNode*)(pmemptr_ + offset);
}

} // namespace ROCKSDB_NAMESPACE