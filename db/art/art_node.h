//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <vector>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

struct InnerNode;

enum ArtNodeType : uint8_t {
  kNode4 = 1,
  kNode16 = 2,
  kNode48 = 3,
  kNode256 = 4,
};

struct ArtNodeHeader{
  ArtNodeType  art_type_      = kNode4;
  uint8_t      prefix_length_ = 0;
  uint16_t     num_children_  = 0;
};

struct ArtNode4 {
  ArtNodeHeader header_;
  unsigned char keys_[4];
  InnerNode *children_[4];

  ArtNode4();
};

struct ArtNode16 {
  ArtNodeHeader header_;
  unsigned char keys_[16];
  InnerNode *children_[16];

  ArtNode16();
};

struct ArtNode48 {
  ArtNodeHeader header_;
  unsigned char keys_[256];
  InnerNode *children_[48];

  ArtNode48();
};

struct ArtNode256 {
  ArtNodeHeader header_;
  InnerNode *children_[256];

  ArtNode256();
};

/* Helper functions */

ArtNodeType ChooseArtNodeType(int size);

ArtNodeHeader *AllocateArtNode(ArtNodeType node_type);

ArtNodeHeader *AllocateArtAfterSplit(
    const std::vector<InnerNode*>&inserted_nodes,
    const std::vector<unsigned char> &c,
    InnerNode *first_node_in_art);

ArtNodeHeader *ReallocateArtNode(ArtNodeHeader *art);

InnerNode *FindChild(InnerNode *node, unsigned char c);

void InsertToArtNode(
    ArtNodeHeader *art, InnerNode *leaf,
    unsigned char c, bool insert_to_group);

} // namespace ROCKSDB_NAMESPACE