//
// Created by joechen on 2022/4/3.
//

#include "art_node.h"

#include <cstring>
#include <cassert>

#include "utils.h"
#include "macros.h"
#include "global_memtable.h"
#include "heat_group.h"

namespace ROCKSDB_NAMESPACE {

ArtNode4::ArtNode4() {
  memset(keys_, 0, 4);
  memset(children_, 0, sizeof(void*) * 4);
}

ArtNode16::ArtNode16() {
  memset(keys_, 0, 16);
  memset(children_, 0, sizeof(void*) * 16);
}

ArtNode48::ArtNode48() {
  memset(keys_, 0, 256);
  memset(children_, 0, sizeof(void*) * 48);
}

ArtNode256::ArtNode256() {
  memset(children_, 0, sizeof(void*) * 256);
}

ArtNodeType ChooseArtNodeType(int size) {
  // precalculate node type
  static int type[256] = {
      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
  };

  return static_cast<ArtNodeType>(type[size - 1]);
}

ArtNodeHeader *AllocateArt4AndInsertNodes(
    const ::std::vector<InnerNode*>& inner_nodes,
    const ::std::vector<unsigned char> &c,
    InnerNode *first_node_in_art) {
  auto *node4 = new ArtNode4();
  for (size_t i = 0; i < c.size(); ++i) {
    node4->children_[i + 1] = inner_nodes[i];
    node4->keys_[i + 1] = c[i];
  }

  node4->children_[0] = first_node_in_art;
  node4->keys_[0] = 0;

  auto header = (ArtNodeHeader *)node4;
  header->num_children_ = inner_nodes.size() + 1;
  header->art_type_ = kNode4;
  return header;
}

ArtNodeHeader *AllocateArt16AndInsertNodes(
    const ::std::vector<InnerNode*>& inner_nodes,
    const ::std::vector<unsigned char> &c,
    InnerNode *first_node_in_art) {
  auto *node16 = new ArtNode16();
  for (size_t i = 0; i < c.size(); ++i) {
    node16->keys_[i + 1] = c[i];
    node16->children_[i + 1] = inner_nodes[i];
  }

  node16->children_[0] = first_node_in_art;
  node16->keys_[0] = 0;

  auto header = (ArtNodeHeader *)node16;
  header->num_children_ = inner_nodes.size() + 1;
  header->art_type_ = kNode16;
  return header;
}

ArtNodeHeader *AllocateArt48AndInsertNodes(
    const ::std::vector<InnerNode*>& inner_nodes,
    const ::std::vector<unsigned char> &c,
    InnerNode *first_node_in_art) {
  auto *node48 = new ArtNode48();
  for (size_t i = 0; i < c.size(); ++i) {
    node48->keys_[c[i]] = i + 1;
    node48->children_[i + 1] = inner_nodes[i];
  }

  node48->children_[0] = first_node_in_art;
  node48->keys_[0] = 1;

  auto header = (ArtNodeHeader *)node48;
  header->num_children_ = inner_nodes.size() + 1;
  header->art_type_ = kNode48;
  return header;
}

ArtNodeHeader *AllocateArt256AndInsertNodes(
    const ::std::vector<InnerNode*>& inner_nodes,
    const ::std::vector<unsigned char> &c,
    InnerNode *first_node_in_art) {
  auto *node256 = new ArtNode256;
  for (size_t i = 0; i < c.size(); ++i) {
    node256->children_[c[i]] = inner_nodes[i];
  }

  node256->children_[0] = first_node_in_art;
  node256->header_.num_children_ = inner_nodes.size() + 1;
  node256->header_.art_type_ = kNode256;
  return (ArtNodeHeader *)node256;
}

ArtNodeHeader *AllocateArtAfterSplit(
    const ::std::vector<InnerNode*>&inserted_nodes,
    const ::std::vector<unsigned char> &c,
    InnerNode *first_node_in_art) {
  size_t num_children = inserted_nodes.size() + 1;
  auto nodeType = ChooseArtNodeType(num_children);

  switch (nodeType) {
    case kNode4:
      return AllocateArt4AndInsertNodes(inserted_nodes, c,
                                        first_node_in_art);
    case kNode16:
      return AllocateArt16AndInsertNodes(inserted_nodes, c,
                                         first_node_in_art);
    case kNode48:
      return AllocateArt48AndInsertNodes(inserted_nodes, c,
                                         first_node_in_art);
    case kNode256:
      return AllocateArt256AndInsertNodes(inserted_nodes, c,
                                          first_node_in_art);
  }

  return nullptr;
}

ArtNodeHeader *AllocateArtNode(ArtNodeType node_type) {
  ArtNodeHeader *node;
  switch (node_type) {
    case kNode4:
      node = (ArtNodeHeader *)(new ArtNode4());
      break;
    case kNode16:
      node = (ArtNodeHeader *)(new ArtNode16());
      break;
    case kNode48:
      node = (ArtNodeHeader *)(new ArtNode48());
      break;
    case kNode256:
      node = (ArtNodeHeader *)(new ArtNode256());
      break;
    default:
      assert(false);
  }
  node->art_type_ = node_type;
  node->prefix_length_ = 0;
  node->num_children_ = 0;
  return node;
}

ArtNodeHeader *ReallocateArtNode16(ArtNodeHeader *art) {
  auto new_art = AllocateArtNode(kNode16);
  auto node16 = (ArtNode16 *)new_art;
  auto node4 = (ArtNode4 *)art;

  memcpy(node4->children_, node16->children_,
         sizeof(void *) * art->num_children_);
  memcpy(node4->keys_, node16->keys_, art->num_children_);
  new_art->num_children_ = art->num_children_;
  new_art->prefix_length_ = art->prefix_length_;
  return new_art;
}

ArtNodeHeader *ReallocateArtNode48(ArtNodeHeader *art) {
  auto new_art = AllocateArtNode(kNode48);
  auto node48 = (ArtNode48 *)new_art;
  auto node16 = (ArtNode16 *)art;

  memcpy(node48->children_, node16->children_,
         sizeof(void *) * art->num_children_);
  for (int i = 0; i < art->num_children_; i++) {
    node48->keys_[node16->keys_[i]] = i + 1;
  }
  new_art->num_children_ = art->num_children_;
  new_art->prefix_length_ = art->prefix_length_;
  return new_art;
}

ArtNodeHeader *ReallocateArtNode256(ArtNodeHeader *art) {
  auto new_art = AllocateArtNode(kNode256);
  auto node256 = (ArtNode256 *)new_art;
  auto node48 = (ArtNode48 *)art;

  for (int i = 0; i < 256; i++) {
    if (node48->keys_[i]) {
      node256->children_[i] = node48->children_[node48->keys_[i] - 1];
    }
  }
  new_art->num_children_ = art->num_children_;
  new_art->prefix_length_ = art->prefix_length_;
  return new_art;
}

ArtNodeHeader *ReallocateArtNode(ArtNodeHeader *art) {
  switch (art->art_type_) {
    case kNode4:
      return ReallocateArtNode16(art);
    case kNode16:
      return ReallocateArtNode48(art);
    case kNode48:
      return ReallocateArtNode256(art);
    default:
      return art;
  }
}

InnerNode *FindChildInNode4(ArtNodeHeader *art, unsigned char c) {
  auto node4 = (ArtNode4 *)art;
  for (int i = 0; i < art->num_children_; i++) {
    if (node4->keys_[i] == c) {
      return node4->children_[i];
    }
  }
  return nullptr;
}

InnerNode *FindChildInNode16(ArtNodeHeader *art, unsigned char c) {
  auto node16 = (ArtNode16 *)art;

  // Copy from
  // https://github.com/armon/libart/blob/master/src/art.c

#ifdef __i386__
  // Compare the key to all 16 stored keys
  __m128i cmp;
  cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                       _mm_loadu_si128((__m128i *)p.p2->keys));

  // Use a mask to ignore children that don't exist
  mask = (1 << n->num_children_) - 1;
  bitfield = _mm_movemask_epi8(cmp) & mask;
#else
#ifdef __amd64__
  // Compare the key to all 16 stored keys
  __m128i cmp;
  cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                       _mm_loadu_si128((__m128i *)node16->keys_));

  // Use a mask to ignore children that don't exist
  int mask = (1 << art->num_children_) - 1;
  int bitfield = _mm_movemask_epi8(cmp) & mask;
#else
  // Compare the key to all 16 stored keys
  bitfield = 0;
  for (i = 0; i < 16; ++i) {
    if (p.p2->keys[i] == c) {
      bitfield |= (1 << i);
    }
  }

  // Use a mask to ignore children that don't exist
  mask = (1 << n->num_children_) - 1;
  bitfield &= mask;
#endif
#endif

  if (bitfield) {
    assert(__builtin_ctz(bitfield) < 16);
  }

  return bitfield ? node16->children_[__builtin_ctz(bitfield)] : nullptr;
}

InnerNode *FindChildInNode48(ArtNodeHeader *art, unsigned char c) {
  auto node48 = (ArtNode48 *)art;
  int pos = node48->keys_[c];
  return pos ? node48->children_[pos - 1] : nullptr;
}

InnerNode *FindChildInNode256(ArtNodeHeader *art, unsigned char c) {
  auto node256 = (ArtNode256 *)art;
  assert(static_cast<int>(c) < 256);
  return node256->children_[static_cast<int>(c)];
}

InnerNode *FindChild(InnerNode *node, unsigned char c) {
  if (IS_LEAF(node->status_)) {
    return nullptr;
  }

  auto art = node->art;
  switch (art->art_type_) {
    case kNode4:
      return FindChildInNode4(art, c);
    case kNode16:
      return FindChildInNode16(art, c);
    case kNode48:
      return FindChildInNode48(art, c);
    case kNode256:
      return FindChildInNode256(art, c);
    default:
      return nullptr;
  }
}

// Insert new leaf node into art, and return prefix node. Prefix node will always exist.
InnerNode *InsertToArtNode4(
    ArtNodeHeader *art, InnerNode *leaf, unsigned char c) {
  auto node4 = (ArtNode4 *)art;
  int idx;
  for (idx = 0; idx < art->num_children_; idx++) {
    if (c < node4->keys_[idx]) {
      break;
    }
  }
  memmove(node4->keys_ + idx + 1,
          node4->keys_ + idx,
          art->num_children_ - idx);
  memmove(node4->children_ + idx + 1, node4->children_ + idx,
          (art->num_children_ - idx) * sizeof(void *));

  // Insert element
  node4->keys_[idx] = c;
  node4->children_[idx] = leaf;
  return idx > 0 ? node4->children_[idx - 1] : nullptr;
}

InnerNode *InsertToArtNode16(
    ArtNodeHeader *art, InnerNode *leaf, unsigned char c) {
  auto node16 = (ArtNode16 *)art;
  unsigned mask = (1 << art->num_children_) - 1;

// support non-x86 architectures
#ifdef __i386__
  __m128i cmp;

  // Compare the key to all 16 stored keys
  cmp = _mm_cmplt_epi8(_mm_set1_epi8(c),
                       _mm_loadu_si128((__m128i *)n->keys));

  // Use a mask to ignore children that don't exist
  unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
#ifdef __amd64__
  __m128i cmp;

  // Compare the key to all 16 stored keys
  cmp = _mm_cmplt_epi8(_mm_set1_epi8(c),
                       _mm_loadu_si128((__m128i *)node16->keys_));

  // Use a mask to ignore children that don't exist
  unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
  // Compare the key to all 16 stored keys
  unsigned bitfield = 0;
  for (short i = 0; i < 16; ++i) {
    if (c < n->keys[i])
      bitfield |= (1 << i);
  }

  // Use a mask to ignore children that don't exist
  bitfield &= mask;
#endif
#endif

  // Check if less than any
  unsigned idx;
  if (bitfield) {
    idx = __builtin_ctz(bitfield);
    memmove(node16->keys_ + idx + 1, node16->keys_ + idx, art->num_children_ - idx);
    memmove(node16->children_ + idx + 1, node16->children_ + idx,
            (art->num_children_ - idx) * sizeof(void *));
  } else {
    idx = art->num_children_;
  }

  // Set the child
  node16->keys_[idx] = c;
  node16->children_[idx] = leaf;
  return idx > 0 ? node16->children_[idx - 1] : nullptr;
}

InnerNode *InsertToArtNode48(
    ArtNodeHeader *art, InnerNode *leaf, unsigned char c) {
  auto node48 = (ArtNode48 *)art;
  int pos = 0;
  while (node48->children_[pos]) {
    pos++;
  }
  node48->children_[pos] = leaf;
  node48->keys_[c] = pos + 1;
  return pos > 0 ? node48->children_[node48->keys_[c] - 1] : nullptr;
}

InnerNode *InsertToArtNode256(
    ArtNodeHeader *art, InnerNode *leaf, unsigned char c) {
  auto node256 = (ArtNode256 *)art;
  int pos = static_cast<int>(c);
  assert(pos < 256);
  node256->children_[pos] = leaf;
  for (int i = c - 1; i >= 0; --i) {
    if (node256->children_[i]) {
      return node256->children_[i];
    }
  }
  return nullptr;
}

void InsertToArtNode(
    ArtNodeHeader *art, InnerNode *leaf, unsigned char c, bool insert_to_group) {
  static int fullNum[5] = {0, 4, 16, 48, 256};

  InnerNode *left_node;
  switch (art->art_type_) {
    case kNode4:
      left_node = InsertToArtNode4(art, leaf, c);
      break;
    case kNode16:
      left_node = InsertToArtNode16(art, leaf, c);
      break;
    case kNode48:
      left_node = InsertToArtNode48(art, leaf, c);
      break;
    case kNode256:
      left_node = InsertToArtNode256(art, leaf, c);
      break;
    default:
      break;
  }

  InsertInnerNode(left_node, leaf);

  if ((++art->num_children_) == fullNum[art->art_type_]) {
    SET_ART_FULL(leaf->status_);
  }

  if (insert_to_group) {
    GroupInsertNewNode(left_node, leaf);
  }
}

} // namespace ROCKSDB_NAMESPACE