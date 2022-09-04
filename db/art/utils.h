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

/////////////////////////////////////////////////////

#ifdef ART_LITTLE_ENDIAN
struct KVStruct {
  union {
    uint64_t hash;
    struct {
      uint32_t actual_hash;
      uint8_t  key_length;
      char     prefixes[3];
    };
  };
  union {
    uint64_t vptr;
    struct {
      uint64_t actual_vptr  : 40;
      uint64_t insert_times : 8;
      int32_t  kv_size      : 16;
    };
  };

  KVStruct() = default;

  KVStruct(uint64_t hash_, uint64_t vptr_) : hash(hash_), vptr(vptr_) {};
};

inline char GetPrefix(const KVStruct& kvInfo, size_t level) {
  assert(level > 0);
  return kvInfo.prefixes[(level - 1) % 3];
}

inline void GetActualVptr(uint64_t& vptr) {
  vptr &= 0x000000ffffffffff;
}

inline void UpdateVptrInfo(uint64_t old_vptr, uint64_t& new_vptr) {
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

inline void UpdateVptrInfo(uint64_t old_vptr, uint64_t& new_vptr) {
  new_vptr <<= 16;
  new_vptr |= (old_vptr & 0x000000000000ffff);
}
inline void UpdateInsertTimes(uint64_t& vptr, uint64_t insert_times) {
  vptr &= 0xffffffffff00ffff;
  vptr |= (insert_times << 16);
}
#endif

/////////////////////////////////////////////////////

inline void HashOnly(KVStruct& s, Slice& key) {
  assert(key.size() < 256);
  memset(s.prefixes, 0, 3);
  s.key_length = key.size();
  s.actual_hash = Hash(key.data(), key.size(), 397);
}

template<typename T>
inline uint64_t HashAndPrefix(T& key, size_t level) {
  KVStruct s{};
  s.key_length = key.size();
  s.actual_hash = Hash(key.data(), key.size(), 397);
  level -= (level - 1) % 3;
  memcpy(s.prefixes, key.data() + level, std::max(level, std::min(key.size(), level + 3)) - level);
  return s.hash;
}

template uint64_t HashAndPrefix<std::string>(std::string&, size_t);
template uint64_t HashAndPrefix<Slice>(Slice&, size_t);

// Maybe this function name is misleading,
// but it just modifies prefix part in hash.
inline void Rehash(KVStruct& s, const Slice& key, size_t level) {
  assert(level > 0);
  level -= (level - 1) % 3;
  memset(s.prefixes, 0, 3);
  memcpy(s.prefixes, key.data() + level,
         std::max(level, std::min(key.size(), level + 3)) - level);
}

inline void Rehash(KVStruct& s, std::string& key, size_t level) {
  assert(level > 0);
  level -= (level - 1) % 3;
  memset(s.prefixes, 0, 3);
  memcpy(s.prefixes, key.data() + level,
         std::max(level, std::min(key.size(), level + 3)) - level);
}

inline bool ActualVptrSame(uint64_t vptr1, uint64_t vptr2) {
  return !((vptr1 ^ vptr2) & 0x000000ffffffffff);
}

int EstimateDistinctCount(const uint8_t hll[64]);

/////////////////////////////////////////////////////
// NVMNode

// We must ensure that next node is not being compacted,
// because nvm_ptr may be changed during compaction.
InnerNode* AllocateLeafNode(uint8_t prefix_length,
                            unsigned char last_prefix,
                            InnerNode* next_node = nullptr,
                            uint64_t init_tag = 0);

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

