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

  NVMNode *AllocateNode();

  void DeallocateNode(char *addr);

  void recoverOnRestart();

  int64_t relative(NVMNode *node);

  NVMNode *absolute(int64_t offset);

 private:
  char *pmemptr_;

  std::atomic<int> num_free_{1048576};

  int64_t total_size_;

  TQueueConcurrent<char *> free_pages_;
};

NodeAllocator &GetNodeAllocator();

////////////////////////////////////////////////

#ifdef ART_LITTLE_ENDIAN
struct KVStruct {
  union {
    uint64_t hash;
    struct {
      char padding[4];
      unsigned char valueType;
      char prefixes[3];
    };
  };
  union {
    uint64_t vptr;
    struct {
      char padding2[6];
      uint16_t kvSize;
    };
  };

  KVStruct() = default;

  KVStruct(uint64_t hash_, uint64_t vptr_) :
                                             hash(hash_), vptr(vptr_) {};
};
#else
struct KVStruct {
  union {
    uint64_t hash;
    char prefixes[4];
  };
  union {
    uint64_t vptr_;
    uint16_t kvSize;
  };
};
#endif

char getPrefix(int level, const KVStruct &kvInfo);

} // namespace ROCKSDB_NAMESPACE