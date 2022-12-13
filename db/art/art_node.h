//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

struct InnerNode;

enum ArtNodeType : uint8_t {
  kNode4 = 1,
  kNode16 = 2,
  kNode48 = 3,
  kNode256 = 4,
};

struct ArtNode{
  ArtNodeType art_type_ = kNode4;
  uint16_t    num_children_ = 0;
};

struct ArtNode4 {
  ArtNode       header_;
  unsigned char keys_[4] = {0};
  InnerNode*    children_[4] = {nullptr};

  ArtNode4() = default;
};

struct ArtNode16 {
  ArtNode       header_;
  unsigned char keys_[16] = {0};
  InnerNode*    children_[16] = {nullptr};

  ArtNode16() = default;
};

struct ArtNode48 {
  ArtNode       header_;
  unsigned char keys_[256] = {0};
  InnerNode*    children_[48] = {nullptr};

  ArtNode48() = default;
};

struct ArtNode256 {
  ArtNode       header_;
  InnerNode*    children_[256] = {nullptr};

  ArtNode256() = default;
};

/* Helper functions */

ArtNodeType ChooseArtNodeType(size_t size);

ArtNode* AllocateArtNode(ArtNodeType node_type);

ArtNode* AllocateArtAfterSplit(
    const std::vector<InnerNode*>& inserted_nodes,
    const std::vector<unsigned char>& c);

ArtNode* ReallocateArtNode(ArtNode* art);

InnerNode* FindChild(InnerNode* node, unsigned char c);

InnerNode* FindChild(InnerNode* node, std::string& key, size_t level,
                     InnerNode** backup, size_t& backup_level);

InnerNode* FindChild(ArtNode* backup, unsigned char c);

void InsertToArtNode(
    InnerNode* current, InnerNode* leaf,
    unsigned char c, bool insert_to_group);

void DeleteInnerNode(InnerNode* inner_node, uint64_t* inode_vptrs, int count);

void DeleteArtNode(ArtNode* art);

// allocated can be negative (e.g. delete nodes)
void UpdateAllocatedSpace(int64_t allocated);

int64_t GetAllocatedSpace();

} // namespace ROCKSDB_NAMESPACE