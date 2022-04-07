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

// Forward declaration
struct InnerNode;
struct NVMNode;
struct ArtNodeHeader;

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

uint64_t Hash(
    const char *key, size_t n,
    [[maybe_unused]] ValueType value_type = kTypeValue);

int EstimateDistinctCount(const uint8_t hyperLogLog[64]);

/////////////////////////////////////////////////////

// Allocate art node and insert inner nodes.
// Inner nodes should be in order.
ArtNodeHeader *AllocateArtAfterSplit(
    const std::vector<InnerNode*>&inserted_nodes,
    const std::vector<unsigned char> &c,
    InnerNode *first_node_in_art);

ArtNodeHeader *ReallocateArtNode(ArtNodeHeader *art);

InnerNode *FindChild(InnerNode *node, unsigned char c);

void InsertToArtNode(ArtNodeHeader *art, InnerNode *leaf,
                     unsigned char c, bool insert_to_group = true);

/////////////////////////////////////////////////////
// NVMNode

InnerNode *AllocateLeafNode(uint8_t prefix_length,
                            unsigned char last_prefix,
                            InnerNode *next_node = nullptr);

// inserted must be initialized
void InsertSplitInnerNode(InnerNode *node, InnerNode *first_inserted,
                          InnerNode *last_inserted, int prefix_length);

// inserted must be initialized
void InsertInnerNode(InnerNode *node, InnerNode *inserted);

// These two functions are used in compaction.
void InsertNewNVMNode(InnerNode *node, NVMNode *new_nvm_node);

void RemoveOldNVMNode(InnerNode *node);

NVMNode* GetNextNode(NVMNode *node);

NVMNode* GetNextNode(int64_t offset);

int64_t GetNextRelativeNode(NVMNode *node);

int64_t GetNextRelativeNode(int64_t offset);
} // namespace ROCKSDB_NAMESPACE

