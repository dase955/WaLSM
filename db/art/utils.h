//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <immintrin.h>
#include <rocksdb/rocksdb_namespace.h>
#include <db/dbformat.h>
#include "macros.h"

namespace ROCKSDB_NAMESPACE {

constexpr size_t RecordPrefixSize
    = 1 + sizeof(SequenceNumber) + sizeof(RecordIndex);

// Forward declaration
struct InnerNode;
struct NVMNode;
struct ArtNodeHeader;

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

uint64_t Hash(
    const char* key, size_t n,
    [[maybe_unused]] ValueType value_type = kTypeValue);

int EstimateDistinctCount(const uint8_t hyperLogLog[64]);

/////////////////////////////////////////////////////

#ifdef ART_LITTLE_ENDIAN
struct KVStruct {
  union {
    uint64_t hash_;
    struct {
      char padding[4];
      unsigned char type_;
      char prefixes_[3];
    };
  };
  union {
    uint64_t vptr_;
    struct {
      char padding2[6];
      uint16_t kv_size_;
    };
  };

  KVStruct() = default;

  KVStruct(uint64_t hash, uint64_t vptr) : hash_(hash), vptr_(vptr) {};
};

inline char GetPrefix(int level, const KVStruct& kvInfo) {
  return kvInfo.prefixes_[level - 2];
}

inline void GetActualVptr(uint64_t& vptr) {
  vptr &= 0x0000ffffffffffff;
}

inline void UpdateActualVptr(uint64_t old_vptr, uint64_t& new_vptr) {
  new_vptr |= (old_vptr & 0xffff000000000000);
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
#endif

/////////////////////////////////////////////////////

// Allocate art node and insert inner nodes.
// Inner nodes should be in order.
ArtNodeHeader* AllocateArtAfterSplit(
    const std::vector<InnerNode*>& inserted_nodes,
    const std::vector<unsigned char>& c,
    InnerNode* first_node_in_art);

ArtNodeHeader* ReallocateArtNode(ArtNodeHeader* art);

void InsertToArtNode(ArtNodeHeader* art, InnerNode* leaf,
                     unsigned char c, bool insert_to_group = true);

/////////////////////////////////////////////////////
// NVMNode

// We must ensure that next node is not being compacted,
// because nvm_ptr may be changed during compaction.
InnerNode* AllocateLeafNode(uint8_t prefix_length,
                            unsigned char last_prefix,
                            InnerNode* next_node = nullptr);

// inserted must be initialized
void InsertSplitInnerNode(InnerNode* node, InnerNode* first_inserted,
                          InnerNode* last_inserted, int prefix_length);

// inserted must be initialized
void InsertInnerNode(InnerNode* node, InnerNode* inserted);

// These two functions are used in compaction.
void InsertNewNVMNode(InnerNode* node, NVMNode* new_nvm_node);

void RemoveOldNVMNode(InnerNode* node);

NVMNode* GetNextNode(NVMNode* node);

NVMNode* GetNextNode(int64_t offset);

int64_t GetNextRelativeNode(NVMNode* node);

int64_t GetNextRelativeNode(int64_t offset);

} // namespace ROCKSDB_NAMESPACE

