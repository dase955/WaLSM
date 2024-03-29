//
// Created by joechen on 22-4-18.
//

#include "nvm_manager.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "utils.h"

char* base_memptr = nullptr;

char* aligned_ptr = nullptr;

int64_t TotalSize = 0;

std::unordered_map<std::string, char*> memories;

namespace ROCKSDB_NAMESPACE {

bool InitializeMemory(std::unordered_map<std::string, int64_t>& memory_usages,
                      const std::string& nvm_path) {
  TotalSize = 4096; // used for alignment
  for (auto& memory_usage : memory_usages) {
    TotalSize += memory_usage.second;
  }

#ifdef USE_PMEM
  int fd;

  struct stat buffer;
  bool file_exist = stat(nvm_path.c_str(), &buffer) == 0;
  if (!file_exist) {
    fd = open(nvm_path.c_str(), O_CREAT|O_RDWR, 0666);
    assert(-1 != fd);
    posix_fallocate(fd, 0, TotalSize);
  } else {
    fd = open(nvm_path.c_str(), O_CREAT, 0666);
    assert(-1 != fd);
  }

  int is_pmem;
  size_t mapped_len;
  base_memptr = (char*)pmem_map_file(
      nvm_path.c_str(), TotalSize, PMEM_FILE_CREATE, 0666, &mapped_len, &is_pmem);
  //assert(is_pmem && mapped_len == (size_t)TotalSize);
  aligned_ptr = reinterpret_cast<char*>(ALIGN_UP(reinterpret_cast<size_t>(base_memptr), 256));

  close(fd);
#else
  int fd;

  struct stat buffer;
  bool file_exist = stat(nvm_path.c_str(), &buffer) == 0;
  if (!file_exist) {
    fd = open(nvm_path.c_str(), O_RDWR|O_CREAT, 00777);
    assert(-1 != fd);
    lseek(fd, TotalSize - 1, SEEK_SET);
    write(fd, "", 1);
  } else {
    fd = open(nvm_path.c_str(), O_RDWR, 00777);
  }

  base_memptr = (char*)mmap(
      nullptr, TotalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  aligned_ptr = reinterpret_cast<char*>(ALIGN_UP(reinterpret_cast<size_t>(base_memptr), 256));
  close(fd);
#endif

  printf("mmap: %p, %p\n", base_memptr, aligned_ptr);

  int64_t offset = 0;
  for (auto& memory_usage : memory_usages) {
    memories.emplace(memory_usage.first, aligned_ptr + offset);
    offset += memory_usage.second;
  }

  return file_exist;
}

char* GetMappedAddress(const std::string& name) {
  return memories[name];
}

void UnmapMemory() {
#ifdef USE_PMEM
  pmem_unmap(base_memptr, TotalSize);
#else
  munmap(base_memptr, TotalSize);
#endif
}

} // namespace ROCKSDB_NAMESPACE
