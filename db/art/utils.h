//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <rocksdb/rocksdb_namespace.h>
#include <cstdint>
#include <immintrin.h>
#include <condition_variable>
#include <util/hash.h>
#include <db/dbformat.h>
#include "macros.h"

#ifdef USE_PMEM
#include <libpmem.h>
#endif

namespace ROCKSDB_NAMESPACE {

struct InnerNode;
struct NVMNode;
struct ArtNode;

#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)

inline uint64_t HashOnly(const char* key, size_t n) {
  assert(n < 256);
  uint32_t prefix = 0;
  ((uint8_t*)&prefix)[0] = (uint8_t)(n & 255);
  return (static_cast<uint64_t>(prefix) << 32) + Hash(key, n, 397);
}

inline uint64_t HashAndPrefix(const char* key, size_t n, size_t level) {
  assert(n < 256);
  assert(level > 0);
  level -= (level - 1) % 3;
  uint32_t prefix = 0;
  memcpy((char*)&prefix + 1, key + level,
         std::max(level, std::min(n, level + 3)) - level);
  ((uint8_t*)&prefix)[0] = (uint8_t)(n & 255);
  return (static_cast<uint64_t>(prefix) << 32) + Hash(key, n, 397);
}

// Maybe this function name is misleading,
// but it just modifies prefix part in hash.
inline void Rehash(const char* key, size_t n, uint64_t& hash, size_t level) {
  assert(level > 0);
  level -= (level - 1) % 3;
  memset((uint8_t*)&hash + 5, 0, 3);
  memcpy((uint8_t*)&hash + 5, key + level,
         std::max(level, std::min(n, level + 3)) - level);
}

int EstimateDistinctCount(const uint8_t hll[64]);

/////////////////////////////////////////////////////

#ifdef ART_LITTLE_ENDIAN
struct KVStruct {
  union {
    uint64_t hash_;
    struct {
      char padding[4];
      uint8_t key_length_;
      char prefixes_[3];
    };
  };
  union {
    uint64_t vptr_;
    struct {
      char padding2[5];
      uint8_t  insert_times;
      uint16_t kv_size_;
    };
  };

  KVStruct() = default;

  KVStruct(uint64_t hash, uint64_t vptr) : hash_(hash), vptr_(vptr) {};
};

inline char GetPrefix(const KVStruct& kvInfo, size_t level) {
  assert(level > 0);
  return kvInfo.prefixes_[(level - 1) % 3];
}

inline void GetActualVptr(uint64_t& vptr) {
  vptr &= 0x000000ffffffffff;
}

inline void UpdateActualVptr(uint64_t old_vptr, uint64_t& new_vptr) {
  new_vptr |= (old_vptr & 0xffffff0000000000);
}

inline int GetInsertTimes(uint64_t& vptr) {
  return (vptr >> 40) & 0x0000ff;
}

inline void UpdateInsertTimes(uint64_t& vptr, uint64_t insert_times) {
  vptr &= 0xffff00ffffffffff;
  vptr |= (insert_times << 40);
}
#else
struct KVStruct {
  union {
    uint64_t hash_;
    char prefixes_[4];
  };
  union {
    uint64_t vptr_;
    uint16_t kv_size_;
  };
};

inline char GetPrefix(int level, const uint64_t &hash) {
  return ((char*)&hash)[level];
}

inline void GetActualVptr(uint64_t& vptr) {
  vptr >>= 16;
}

inline void UpdateActualVptr(uint64_t old_vptr, uint64_t& new_vptr) {
  new_vptr <<= 16;
  new_vptr |= (old_vptr & 0x000000000000ffff);
}
inline void UpdateInsertTimes(uint64_t& vptr, uint64_t insert_times) {
  vptr &= 0xffffffffff00ffff;
  vptr |= (insert_times << 16);
}
#endif

/////////////////////////////////////////////////////

// Allocate art node and insert inner nodes.
// Inner nodes should be in order.
ArtNode* AllocateArtAfterSplit(
    const std::vector<InnerNode*>& inserted_nodes,
    const std::vector<unsigned char>& c,
    InnerNode* first_node_in_art);

/////////////////////////////////////////////////////
// NVMNode

// We must ensure that next node is not being compacted,
// because nvm_ptr may be changed during compaction.
InnerNode* AllocateLeafNode(uint8_t prefix_length,
                            unsigned char last_prefix,
                            InnerNode* next_node = nullptr);

InnerNode* RecoverInnerNode(NVMNode* nvm_node);

// inserted must be initialized
void InsertSplitInnerNode(InnerNode* node, InnerNode* first_inserted,
                          InnerNode* last_inserted, size_t prefix_length);

// inserted must be initialized
void InsertInnerNode(InnerNode* node, InnerNode* inserted);

// These two functions are used in compaction.
void InsertNewNVMNode(InnerNode* node, NVMNode* inserted);

// Different from RemoveChildrenNVMNode,
// this function is used to remove backup node
void RemoveOldNVMNode(InnerNode* node);

NVMNode* GetNextNode(NVMNode* node);

NVMNode* GetNextNode(int64_t offset);

int64_t GetNextRelativeNode(NVMNode* node);

int64_t GetNextRelativeNode(int64_t offset);

/////////////////////////////////////////////////////

class BackgroundThread {
 public:
  virtual ~BackgroundThread() = default;

  void StartThread() {
    thread_stop_ = false;
    background_thread_ = std::thread(&BackgroundThread::BGWork, this);
  }

  void StopThread() {
    thread_stop_ = true;
    cond_var_.notify_one();
    background_thread_.join();
    printf("background thread stop.\n");
  }

 private:
  virtual void BGWork() = 0;

 protected:
  std::thread background_thread_;

  std::mutex mutex_;

  std::condition_variable cond_var_;

  bool thread_stop_ = false;
};

} // namespace ROCKSDB_NAMESPACE

