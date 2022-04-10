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

  int fd = open("/tmp/NodeMemory", O_RDWR|O_CREAT, 00777);
  assert(-1 != fd);
  lseek(fd, total_size_ - 1, SEEK_END);
  write(fd, "", 1);
  pmemptr_ = (char *)mmap(NULL, total_size_, PROT_READ | PROT_WRITE,MAP_SHARED, fd, 0);
  close(fd);

  if (recover) {
    recoverOnRestart();
    return;
  }

  char *cur_ptr = pmemptr_;
  for (int i = 0; i < num_free_; ++i) {
    free_pages_.emplace_back(cur_ptr);
    cur_ptr += PAGE_SIZE;
  }
}

NodeAllocator::~NodeAllocator() {
  munmap(pmemptr_, total_size_);
}

NVMNode* NodeAllocator::AllocateNode() {
  char* addr = free_pages_.pop_front();
  return (NVMNode*)addr;
}

void NodeAllocator::DeallocateNode(char* addr) {
  free_pages_.emplace_back(addr);
}

void NodeAllocator::recoverOnRestart() {

}

int64_t NodeAllocator::relative(NVMNode* node) {
  return (char*)node - pmemptr_;
}

NVMNode* NodeAllocator::absolute(int64_t offset) {
  return offset == -1 ? nullptr : (NVMNode*)(pmemptr_ + offset);
}

#ifdef ART_LITTLE_ENDIAN
char GetPrefix(int level, const KVStruct& kvInfo) {
  return kvInfo.prefixes_[level - 2];
}
#else
inline char GetPrefix(int level, const uint64_t &hash) {
  return ((char*)&hash)[level];
}
#endif

} // namespace ROCKSDB_NAMESPACE