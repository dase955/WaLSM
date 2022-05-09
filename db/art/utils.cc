//
// Created by joechen on 2022/4/3.
//

#include "utils.h"

#include <functional>
#include <cstring>
#include <cmath>

#include "art_node.h"
#include "macros.h"
#include "nvm_node.h"
#include "node_allocator.h"
#include "global_memtable.h"

namespace ROCKSDB_NAMESPACE {

int EstimateDistinctCount(const uint8_t hll[64]) {
  static constexpr float alpha = 0.709 * 64 * 64;
  static float factor[32] = {
      1.0f, 0.5f,
      0.25f, 0.125f,
      0.0625f, 0.03125f,
      0.015625f, 0.0078125f,
      0.00390625f, 0.001953125f,
      0.0009765625f, 0.00048828125f,
      0.000244140625f, 0.0001220703125f,
      6.103515625e-05f, 3.0517578125e-05f,
      1.52587890625e-05f, 7.62939453125e-06f,
      3.814697265625e-06f, 1.9073486328125e-06f,
      9.5367431640625e-07f, 4.76837158203125e-07f,
      2.384185791015625e-07f, 1.1920928955078125e-07f,
      5.960464477539063e-08f, 2.9802322387695312e-08f,
      1.4901161193847656e-08f, 7.450580596923828e-09f,
      3.725290298461914e-09f, 1.862645149230957e-09f,
      9.313225746154785e-10f, 4.656612873077393e-10f,
  };

  float sum = 0.f;
  int empty_buckets = 0;
  for (int i = 0; i < 64; i++) {
    sum += factor[hll[i]];
    empty_buckets += (hll[i] == 0);
  }

  float estimated = alpha / sum;
  if (estimated < 160.f && empty_buckets > 0) {
    estimated = std::min(64.f * (float)std::log(64.0 / empty_buckets),
                         estimated);
  }
  return (int)estimated;
}

////////////////////////////////////////////////////////////////////////////

InnerNode* AllocateLeafNode(uint8_t prefix_length,
                            unsigned char last_prefix,
                            InnerNode* next_node) {
  auto inode = new InnerNode();
  uint32_t status = inode->status_;
  SET_LEAF(status);
  SET_NON_GROUP_START(status);
  SET_ART_NON_FULL(status);
  SET_NODE_BUFFER_SIZE(status, 0);
  SET_GC_FLUSH_SIZE(status, 0);
  inode->status_ = status;

  auto mgr = GetNodeAllocator();

  auto nvm_node = mgr->AllocateNode();
  inode->nvm_node_ = nvm_node;
  inode->last_child_node_ = inode;
  inode->next_node_ = next_node;

  uint64_t hdr = 0;
  SET_LAST_PREFIX(hdr, last_prefix);
  SET_LEVEL(hdr, prefix_length);
  SET_TAG(hdr, VALID_TAG);
  SET_TAG(hdr, ALT_FIRST_TAG);

  nvm_node->meta.header = hdr;
  nvm_node->meta.next1 = next_node ? mgr->relative(next_node->nvm_node_) : -1;

  return inode;
}

InnerNode* RecoverInnerNode(NVMNode* nvm_node) {
  auto inode = new InnerNode();

  memcpy(inode->buffer_, nvm_node->temp_buffer, 256);
  memset(nvm_node->temp_buffer, 0, 256);
  int buffer_size = 0;
  for (; buffer_size < 16; ++buffer_size) {
    if (inode->buffer_[buffer_size * 2] == 0) {
      break;
    }
  }

  int h, bucket;
  uint8_t digit;
  int size = GET_SIZE(nvm_node->meta.header);
  for (int i = 0; i < size; ++i) {
    auto vptr = nvm_node->data[i * 2 + 1];
    if (!vptr) {
      continue;
    }

    auto hash = nvm_node->data[i * 2];
    bucket = (int)(hash & 63);
    h = (int)(hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    inode->hll_[bucket] = std::max(inode->hll_[bucket], digit);
  }

  uint32_t status = 0;
  SET_LEAF(status);
  SET_NON_GROUP_START(status);
  SET_ART_NON_FULL(status);
  SET_NODE_BUFFER_SIZE(status, buffer_size);
  SET_GC_FLUSH_SIZE(status, 0);

  inode->status_ = status;
  inode->estimated_size_ = nvm_node->meta.node_info;
  inode->last_child_node_ = inode;
  inode->nvm_node_ = nvm_node;
  inode->backup_nvm_node_ = nullptr;
  return inode;
}

ArtNode* RemoveChildrenNVMNode(InnerNode* parent) {
  // assert parent lock is held, so last child node is unchanged
  InnerNode* last_child_node = parent->last_child_node_;
  std::lock_guard<SpinMutex> link_lk(last_child_node->link_lock_);

  InnerNode* next_inner_node = last_child_node->next_node_;
  [[maybe_unused]] InnerNode* first_remove_node = parent->next_node_;
  parent->next_node_ = next_inner_node;

  NVMNode* cur_nvm_node = parent->nvm_node_;
  NVMNode* next_nvm_node = next_inner_node->nvm_node_;
  int64_t next_relative = GetNodeAllocator()->relative(next_nvm_node);

  auto old_hdr = cur_nvm_node->meta.header;
  auto new_hdr = old_hdr;
  if (GET_TAG(old_hdr, ALT_FIRST_TAG)) {
    CLEAR_TAG(new_hdr, ALT_FIRST_TAG);
    cur_nvm_node->meta.next2 = next_relative;
  } else {
    SET_TAG(new_hdr, ALT_FIRST_TAG);
    cur_nvm_node->meta.next1 = next_relative;
  }
  SET_ROWS(new_hdr, 0);
  SET_SIZE(new_hdr, 0);
  cur_nvm_node->meta.header = new_hdr;
  PERSIST(cur_nvm_node, 8);

  auto art = parent->art;
  delete art->backup_;
  parent->art = nullptr;
  auto status = parent->status_;
  SET_LEAF(status);
  SET_NON_GROUP_START(status);
  SET_ART_NON_FULL(status);
  SET_NODE_BUFFER_SIZE(status, 0);
  SET_GC_FLUSH_SIZE(status, 0);
  parent->status_ = status;
  return art;
}

void InsertInnerNode(InnerNode* node, InnerNode* inserted) {
  std::lock_guard<SpinMutex> link_lk(node->link_lock_);

  auto prev_node = node->last_child_node_;
  inserted->next_node_ = prev_node->next_node_;
  prev_node->next_node_ = inserted;

  auto prev_nvm_node = prev_node->nvm_node_;
  auto inserted_nvm_node = inserted->nvm_node_;
  uint64_t hdr = prev_nvm_node->meta.header;

  // Because we only change pointer here,
  // so there is no need to switch alt bit.
  if (GET_TAG(hdr, ALT_FIRST_TAG)) {
    inserted_nvm_node->meta.next1 = prev_nvm_node->meta.next1;
    prev_nvm_node->meta.next1 =
        GetNodeAllocator()->relative(inserted_nvm_node);
  } else {
    inserted_nvm_node->meta.next1 = prev_nvm_node->meta.next2;
    prev_nvm_node->meta.next2 =
        GetNodeAllocator()->relative(inserted_nvm_node);
  }
  PERSIST(inserted_nvm_node, CACHE_LINE_SIZE);
  PERSIST(prev_nvm_node, CACHE_LINE_SIZE);
}

void InsertSplitInnerNode(InnerNode* node, InnerNode* first_inserted,
                          InnerNode* last_inserted,
                          [[maybe_unused]] size_t prefix_length) {
  std::lock_guard<SpinMutex> lk(node->link_lock_);

  auto prev_node = node->last_child_node_;
  auto prev_nvm_node = prev_node->nvm_node_;
  auto inserted_first_nvm_node = first_inserted->nvm_node_;
  auto inserted_last_nvm_node = last_inserted->nvm_node_;

  auto old_hdr = prev_nvm_node->meta.header;
  auto new_hdr = old_hdr;
  if (GET_TAG(old_hdr, ALT_FIRST_TAG)) {
    CLEAR_TAG(new_hdr, ALT_FIRST_TAG);
    prev_nvm_node->meta.next2 =
        GetNodeAllocator()->relative(inserted_first_nvm_node);
    inserted_last_nvm_node->meta.next1 = prev_nvm_node->meta.next1;
  } else {
    SET_TAG(new_hdr, ALT_FIRST_TAG);
    prev_nvm_node->meta.next1 =
        GetNodeAllocator()->relative(inserted_first_nvm_node);
    inserted_last_nvm_node->meta.next1 = prev_nvm_node->meta.next2;
  }
  PERSIST(inserted_last_nvm_node, CACHE_LINE_SIZE);

  /*SET_LAST_PREFIX(new_hdr, 0);
  SET_LEVEL(new_hdr, prefix_length);
  SET_SIZE(new_hdr, 0);
  SET_ROWS(new_hdr, 0);*/
  CLEAR_TAG(new_hdr, VALID_TAG);
  prev_nvm_node->meta.header = new_hdr;
  PERSIST(prev_nvm_node, CACHE_LINE_SIZE);

  node->estimated_size_ = 0;
  last_inserted->next_node_ = prev_node->next_node_;
  prev_node->next_node_ = first_inserted;

  // Update last child node
  node->last_child_node_ = last_inserted;
}

void InsertNewNVMNode(InnerNode* node, NVMNode* inserted) {
  auto old_nvm_node = node->nvm_node_;
  auto inserted_hdr = old_nvm_node->meta.header;
  SET_TAG(inserted_hdr, ALT_FIRST_TAG);
  SET_ROWS(inserted_hdr, 0);
  SET_SIZE(inserted_hdr, 0);

  {
    std::lock_guard<SpinMutex> link_lk(node->link_lock_);

    node->backup_nvm_node_ = old_nvm_node;
    node->nvm_node_ = inserted;

    auto old_hdr = old_nvm_node->meta.header;
    auto new_hdr = old_hdr;

    if (GET_TAG(old_hdr, ALT_FIRST_TAG)) {
      inserted->meta.next1 = old_nvm_node->meta.next1;
      old_nvm_node->meta.next1 = GetNodeAllocator()->relative(inserted);
    } else {
      inserted->meta.next1 = old_nvm_node->meta.next2;
      old_nvm_node->meta.next2 = GetNodeAllocator()->relative(inserted);
    }

    inserted->meta.header = inserted_hdr;
    PERSIST(inserted, CACHE_LINE_SIZE);

    old_nvm_node->meta.header = new_hdr;
    PERSIST(old_nvm_node, CACHE_LINE_SIZE);
  }
}

// Different from InsertNewNVMNode, link_lock_ has been held
void RemoveOldNVMNode(InnerNode* node) {
  auto next_node = node->next_node_;
  auto nvm_node = node->nvm_node_;
  auto hdr = nvm_node->meta.header;

  if (GET_TAG(hdr, ALT_FIRST_TAG)) {
    nvm_node->meta.next1 = GetNextRelativeNode(nvm_node->meta.next1);
  } else {
    nvm_node->meta.next2 = GetNextRelativeNode(nvm_node->meta.next2);
  }

  PERSIST(nvm_node, CACHE_LINE_SIZE);

  auto backup_nvm_node = next_node->backup_nvm_node_;
  next_node->backup_nvm_node_ = nullptr;
  GetNodeAllocator()->DeallocateNode(backup_nvm_node);
}

void RemoveCompactedNodes(std::vector<InnerNode*>& inner_nodes) {
  // remove parent's children from linked list
  // and store art, these art nodes will be free in next time
  std::vector<ArtNode*> removed_art;

  for (auto inner_node : inner_nodes) {
    inner_node->opt_lock_.WriteLock();
  }

  for (int iter = 0; iter < 2; ++iter) {
    std::unordered_map<InnerNode*, int> parents_and_counts;
    std::vector<InnerNode*> next_inner_nodes;
    for (auto inner_node : inner_nodes) {
      ++parents_and_counts[inner_node->parent_node_];
    }

    bool next_iteration = true;
    for (auto& pair : parents_and_counts) {
      auto parent = pair.first;
      // lock parent
      parent->opt_lock_.WriteLock();
      if (parent->art->num_children_ == pair.second) {
        parent->opt_lock_.WriteUnlock(true);
        next_iteration = false;
        continue;
      }
      next_inner_nodes.push_back(parent);
      removed_art.push_back(RemoveChildrenNVMNode(parent));
    }

    for (auto inner_node : inner_nodes) {
      inner_node->opt_lock_.WriteUnlock(false);
    }

    inner_nodes = next_inner_nodes;
    if (!next_iteration) {
      break;
    }
  }

  for (auto inner_node : inner_nodes) {
    inner_node->opt_lock_.WriteUnlock(false);
  }
}

NVMNode* GetNextNode(NVMNode* node) {
  int64_t next_offset =
      GET_TAG(node->meta.header, ALT_FIRST_TAG)
          ? node->meta.next1 : node->meta.next2;
  return next_offset == -1 ? nullptr :
                           GetNodeAllocator()->absolute(next_offset);
}

NVMNode* GetNextNode(int64_t offset) {
  NVMNode* node = GetNodeAllocator()->absolute(offset);
  int64_t next_offset =
      GET_TAG(node->meta.header, ALT_FIRST_TAG)
          ? node->meta.next1 : node->meta.next2;
  return GetNodeAllocator()->absolute(next_offset);
}

int64_t GetNextRelativeNode(NVMNode* node) {
  return GET_TAG(node->meta.header, ALT_FIRST_TAG)
             ? node->meta.next1 : node->meta.next2;
}

int64_t GetNextRelativeNode(int64_t offset) {
  NVMNode* node = GetNodeAllocator()->absolute(offset);
  return GET_TAG(node->meta.header, ALT_FIRST_TAG)
             ? node->meta.next1 : node->meta.next2;
}
} // namespace ROCKSDB_NAMESPACE