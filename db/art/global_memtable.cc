//
// Created by joechen on 2022/4/3.
//

#include "global_memtable.h"

#include <iostream>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include <db/write_batch_internal.h>

#include "utils.h"
#include "nvm_node.h"
#include "art_node.h"
#include "node_allocator.h"
#include "vlog_manager.h"
#include "heat_group.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

#define SQUEEZE_THRESHOLD 112

InnerNode::InnerNode()
    : heat_group_(nullptr), art(nullptr), artBackup(nullptr),
      nvm_node_(nullptr), backup_nvm_node_(nullptr), last_child_node_(nullptr),
      next_node_(nullptr), vptr_(0), status_(0), estimated_size_(0),
      buffer_size_(0) {
  memset(hll_, 0, 64);
  memset(buffer_, 0, 256);
}

void FlushBuffer(InnerNode *leaf, int row) {
  NVMNode *node = leaf->nvm_node_;
  uint64_t *buffer = leaf->buffer_;
  uint8_t *hll = leaf->hll_;

  int h, bucket;
  uint8_t digit;
  uint8_t fingerprints[16];
  for (size_t i = 0; i < 16; ++i) {
    auto hash = buffer[i << 1];
    fingerprints[i] = static_cast<uint8_t>(hash);

    bucket = hash & 63;
    h = (int) (hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    hll[bucket] = std::max(hll[bucket], digit);
  }

  memcpy(node->data + (row << 5), buffer, 256);
  memcpy(node->meta.fingerprints_ + (row << 4), fingerprints, 16);
  // _mm_clwb(data); _mm_sfence();

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, (size + 16));
  SET_ROWS(hdr, (row + 1));
  node->meta.header = hdr;
  // _mm_clwb(node->meta.header); _mm_sfence();
}

//////////////////////////////////////////////////////////

void GlobalMemtable::InitFirstTwoLevel() {
  root_ = new InnerNode();
  tail_ = new InnerNode();
  SET_NON_LEAF(root_->status_);
  SET_ART_FULL(root_->status_);

  // First level
  root_->art = AllocateArtNode(kNode256);
  auto art256 = (ArtNode256 *)root_->art;
  art256->header_.prefix_length_ = 1;
  art256->header_.num_children_ = 256;

  InnerNode *nextInnerNode = tail_;
  for (int i = LAST_CHAR; i >= 0; --i) {
    auto *node = new InnerNode();
    SET_NON_LEAF(node->status_);
    SET_ART_FULL(node->status_);
    art256->children_[i] = node;

    // Second level
    node->art = AllocateArtNode(kNode256);
    auto art256_2 = (ArtNode256 *)node->art;
    art256_2->header_.prefix_length_ = 2;
    art256_2->header_.num_children_ = 256;

    auto heatGroup = NewHeatGroup();

    for (int j = LAST_CHAR; j >= 0; --j) {
      art256_2->children_[j] = AllocateLeafNode(
          2, static_cast<unsigned char>(j), nextInnerNode);
      art256_2->children_[j]->heat_group_ = heatGroup;
      nextInnerNode = art256_2->children_[j];
    }

    heatGroup->first_node_ = art256_2->children_[0];
    heatGroup->last_node_ = art256_2->children_[LAST_CHAR];
    InsertIntoLayer(heatGroup, 0);
  }

  head_ = nextInnerNode;
}

void GlobalMemtable::Put(Slice &slice, uint64_t vptr) {
  Slice key;
  static constexpr size_t record_prefix =
      WriteBatchInternal::kHeader + 1 + sizeof(SequenceNumber);
  slice.remove_prefix(record_prefix);
  GetLengthPrefixedSlice(&slice, &key);
  auto hash = Hash(key.data(), key.size());

  KVStruct kvInfo(hash, vptr);
  kvInfo.kvSize = slice.size();
  kvInfo.valueType = 0;
  size_t maxDepth = key.size();

  bool needRestart;
  size_t level = 0;
  uint32_t version;
  InnerNode *current = root_;
  InnerNode *nextNode;

  while (true) {
  Restart:
    needRestart = false;
    version = current->opt_lock_.AwaitNodeUnlocked();
    nextNode = FindChild(current, key[level]);
    current->opt_lock_.CheckOrRestart(version, needRestart);
    if (needRestart) {
      goto Restart;
    }

    if (IS_LEAF(current->status_)) {
      current->opt_lock_.UpgradeToWriteLockOrRestart(version, needRestart);
      if (needRestart) {
        goto Restart;
      }

      InsertIntoLeaf(current, kvInfo, level);
      return;
    }

    if (!nextNode) {
      current->opt_lock_.UpgradeToWriteLockOrRestart(version, needRestart);
      if (needRestart) {
        goto Restart;
      }

      InnerNode *leaf = AllocateLeafNode(level + 1, key[level], nullptr);
      ++leaf->status_;
      leaf->ts.UpdateHeat();
      leaf->estimated_size_ += kvInfo.kvSize;
      leaf->buffer_size_ += kvInfo.kvSize;
      leaf->buffer_[0] = kvInfo.hash;
      leaf->buffer_[1] = kvInfo.vptr;

      if (IS_ART_FULL(current->status_)) {
        auto artNew = ReallocateArtNode(current->art);
        SET_ART_NON_FULL(current->status_);
        auto last = current->art;
        current->art = artNew;
        delete current->artBackup;
        current->artBackup = last;
      }

      InsertToArtNode(current->art, leaf, key[level], true);
      current->opt_lock_.WriteUnlock();
      return;
    }

    // level > maxDepth means key need to be stored in vptr_
    // TODO: how to persistence data stored in vptr_
    current = nextNode;
    if (level++ == maxDepth) {
      current->vptr_ = kvInfo.vptr;
      return;
    }
  }
}

bool GlobalMemtable::Get(std::string &key, std::string &value) {
  size_t maxDepth = key.length();

  size_t level = 0;
  InnerNode *current = root_;
  InnerNode *nextNode;

  while (true) {
    nextNode = FindChild(current, key[level]);
    if (IS_LEAF(current->status_)) {
      return FindKey(current, key, value);
    }

    if (!nextNode) {
      return false;
    }

    // level > maxDepth means key need to be stored in vptr_
    // TODO: how to persistence data stored in vptr_
    current = nextNode;
    if (level++ == maxDepth) {
      if (current->vptr_ == 0) {
        return false;
      }

      SequenceNumber dummy_seq_num;
      vlog_manager_->GetKeyValue(current->vptr_, key, value, dummy_seq_num);
      return true;
    }
  }
}

void GlobalMemtable::SetVLogManager(VLogManager *vlog_manager) {
  vlog_manager_ = vlog_manager;
}

void GlobalMemtable::SqueezeNode(InnerNode *leaf) {
  auto node = leaf->nvm_node_;
  uint64_t *data = node->data;
  uint8_t *fingerprints = node->meta.fingerprints_;
  std::unordered_map<std::string, KVStruct> map;
  SequenceNumber dummy_seq_num;

  for (size_t i = 0; i < NVM_MAX_SIZE; i++) {
    uint64_t hash = data[(i << 1)];
    uint64_t vptr = data[(i << 1) + 1];
    if (!vptr) {
      continue;
    }
    std::string key, value;
    vlog_manager_->GetKeyValue(vptr, key, value, dummy_seq_num);
    map[key] = KVStruct(hash, vptr);
  }

  int count = 0, fpos = 0;
  auto prevSize = leaf->estimated_size_;
  int32_t curSize = 0;
  memset(fingerprints, 0, 224);
  for (auto &pair: map) {
    auto &key = pair.first;
    auto &kvInfo = pair.second;
    fingerprints[fpos++] = static_cast<uint8_t>(kvInfo.hash);
    data[count++] = kvInfo.hash;
    data[count++] = kvInfo.vptr;
    curSize += kvInfo.kvSize;
  }

  assert(fpos <= NVM_MAX_SIZE);
  assert(count <= (NVM_MAX_SIZE * 2));

  leaf->estimated_size_ = curSize;
  leaf->heat_group_->updateSize(curSize - prevSize);

  int rows = ((fpos - 1) >> 4) + 1;
  int size = rows << 4;
  memset(data + count, 0, (size - fpos) << 4);
  // _mm_clwb(data);
  // _mm_sfence();

  uint64_t hdr = node->meta.header;
  SET_SIZE(hdr, size);
  SET_ROWS(hdr, rows);
  node->meta.header = hdr;
  // _mm_clwb(&node->meta);
  // _mm_sfence();
}

void GlobalMemtable::SplitLeafBelowLevel5(InnerNode *leaf) {
  if (EstimateDistinctCount(leaf->hll_) < SQUEEZE_THRESHOLD) {
    leaf->opt_lock_.WriteUnlock(false);
    SqueezeNode(leaf);
    return;
  }

  auto node = leaf->nvm_node_;
  auto *data = node->data;
  std::vector<KVStruct> split_buckets[256];

  int level = GET_PRELEN(node->meta.header);
  for (size_t i = 0; i < NVM_MAX_SIZE; ++i) {
    KVStruct kvInfo(data[(i << 1)], data[(i << 1) + 1]);
    split_buckets[static_cast<unsigned int>(getPrefix(level, kvInfo))].push_back(kvInfo); // ??
  }

  Timestamps curTs = leaf->ts.copy();

  int32_t split_num = 2;
  for (size_t i = 1; i < LAST_CHAR; ++i) {
    if (!split_buckets[i].empty()) {
      ++split_num;
    }
  }

  auto final_node = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), leaf->next_node_);
  auto last_node = final_node;
  auto first_node = final_node;
  final_node->ts = curTs;

  std::vector<InnerNode*> new_leaves{final_node};
  std::vector<unsigned char> prefixes{LAST_CHAR};
  new_leaves.reserve(split_num);

  for (int c = LAST_CHAR - 1; c > 0; --c) {
    if (split_buckets[c].empty()) {
      continue;
    }
    auto new_leaf = AllocateLeafNode(level + 1, static_cast<unsigned char>(c), last_node);
    new_leaf->ts = curTs;
    auto nvm_node = new_leaf->nvm_node_;
    int pos = 0, fpos = 0;
    uint32_t leaf_size = 0;
    for (auto &kvInfo : split_buckets[c]) {
      nvm_node->meta.fingerprints_[fpos++] = static_cast<uint8_t>(kvInfo.hash);
      nvm_node->data[pos++] = kvInfo.hash;
      nvm_node->data[pos++] = kvInfo.vptr;
      leaf_size += kvInfo.kvSize;
    }
    new_leaf->estimated_size_ = leaf_size;
    assert(fpos <= NVM_MAX_SIZE);
    assert(pos <= (NVM_MAX_SIZE << 1));

    last_node = new_leaf;
    first_node = new_leaf;

    int rows = ((fpos - 1) >> 4) + 1;
    int size = rows << 4;
    auto hdr = nvm_node->meta.header;
    SET_SIZE(hdr, size);
    SET_ROWS(hdr, rows);
    SET_LAST_PREFIX(hdr, c);
    SET_PRELEN(hdr, level + 1);
    nvm_node->meta.header = hdr;
    // _mm_clwb(nvm_node_->meta); _mm_clwb(hdr);

    assert(rows <= NVM_MAX_ROWS);
    assert(size <= NVM_MAX_SIZE);

    new_leaves.push_back(new_leaf);
    prefixes.push_back(static_cast<unsigned char>(c));
  }

  std::reverse(new_leaves.begin(), new_leaves.end());
  std::reverse(prefixes.begin(), prefixes.end());
  auto art = AllocateArtAfterSplit(new_leaves, prefixes, leaf);
  art->prefix_length_ = level + 1;

  leaf->art = art;
  leaf->vptr_ = split_buckets[0].empty() ? 0 : split_buckets[0].back().vptr;
  SET_ART_NON_FULL(leaf->status_);

  InsertSplitInnerNode(leaf, first_node, final_node, (level + 1));
  // _mm_sfence(); clwb(node->meta); _mm_sfence();
  leaf->estimated_size_ = 0;
  GroupInsertNewNode(leaf, new_leaves);

  SET_NON_LEAF(leaf->status_);
  leaf->opt_lock_.WriteUnlock(false);
}

void GlobalMemtable::SplitNode(InnerNode *oldLeaf) {
  auto node = oldLeaf->nvm_node_;

  auto *data = node->data;
  std::vector<KVStruct> splitBuckets[256];
  std::unordered_set<std::string> uniqueSet;
  uint64_t tmpVptr = 0;

  // TODO: remove duplication
  int level = GET_PRELEN(node->meta.header);
  uint8_t p = GET_LAST_PREFIX(node->meta.header);
  for (int i = NVM_MAX_SIZE - 1; i >= 0; --i) {
    uint64_t hash = data[(i << 1)];
    uint64_t vptr = data[(i << 1) + 1];
    if (!vptr) {
      continue;
    }
    std::string key, value;
    SequenceNumber dummy_seq_num;
    vlog_manager_->GetKeyValue(vptr, key, value, dummy_seq_num);
    if (uniqueSet.find(key) == uniqueSet.end()) {
      splitBuckets[static_cast<unsigned char>(key[level])].emplace_back(hash, vptr);
      uniqueSet.insert(key);
    }
  }

  if (uniqueSet.size() < SQUEEZE_THRESHOLD) {
    oldLeaf->opt_lock_.WriteUnlock(false);
    SqueezeNode(oldLeaf);
    return;
  }

  oldLeaf->vptr_ = splitBuckets[0].empty() ? 0 : splitBuckets[0].back().vptr;

  // First and last_insert_ node is always created
  int32_t splitNum = 2;
  for (size_t i = 1; i < 255; ++i) {
    if (!splitBuckets[i].empty()) {
      ++splitNum;
    }
  }

  auto finalInnerNode = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), nullptr);
  auto lastInnerNode = finalInnerNode;
  auto firstInnerNode = finalInnerNode;

  std::vector<InnerNode*> newInnerNodes{finalInnerNode};
  std::vector<unsigned char> prefixes{LAST_CHAR};
  newInnerNodes.reserve(256);

  for (int c = LAST_CHAR - 1; c > 0; --c) {
    if (splitBuckets[c].empty()) {
      continue;
    }
    auto allocatedLeaf = AllocateLeafNode(level + 1, static_cast<unsigned char>(c), lastInnerNode);
    auto nvmNode = allocatedLeaf->nvm_node_;
    int pos = 0, fpos = 0;
    uint32_t nodeSize = 0;
    for (auto &kvInfo : splitBuckets[c]) {
      nvmNode->meta.fingerprints_[fpos++] = static_cast<uint8_t>(kvInfo.hash);
      nvmNode->data[pos++] = kvInfo.hash;
      nvmNode->data[pos++] = kvInfo.vptr;
      nodeSize += kvInfo.kvSize;
    }
    allocatedLeaf->estimated_size_ = nodeSize;

    lastInnerNode = allocatedLeaf;
    firstInnerNode = allocatedLeaf;

    int rows = ((fpos - 1) >> 4) + 1;
    int size = rows << 4;
    auto hdr = nvmNode->meta.header;
    SET_SIZE(hdr, size);
    SET_ROWS(hdr, rows);
    SET_LAST_PREFIX(hdr, c);
    SET_PRELEN(hdr, level + 1);
    nvmNode->meta.header = hdr;
    // _mm_clwb(nvm_node_->meta); _mm_clwb(hdr);

    newInnerNodes.push_back(allocatedLeaf);
    prefixes.push_back(static_cast<unsigned char>(c));
  }

  std::reverse(newInnerNodes.begin(), newInnerNodes.end());
  std::reverse(prefixes.begin(), prefixes.end());
  auto art = AllocateArtAfterSplit(newInnerNodes, prefixes, oldLeaf);
  art->prefix_length_ = (uint8_t)(level + 1);

  oldLeaf->art = art;
  SET_ART_NON_FULL(oldLeaf->status_);

  InsertSplitInnerNode(oldLeaf, firstInnerNode, finalInnerNode, (level + 1));
  // _mm_sfence(); clwb(node->meta); _mm_sfence();
  GroupInsertNewNode(oldLeaf, newInnerNodes);

  SET_NON_LEAF(oldLeaf->status_);
  oldLeaf->opt_lock_.WriteUnlock(false);
}

// "4b prefix + 4b hash + 8b pointer" OR "4b SeqNum + 4b hash + 8b pointer"
// This function is responsible for unlocking opt_lock_
void GlobalMemtable::InsertIntoLeaf(InnerNode *leaf, KVStruct &kvInfo, int level) {
  ++leaf->status_;

  int writePos = GET_NODE_BUFFER_SIZE(leaf->status_) << 1;
  assert(writePos <= 32);

  leaf->buffer_[writePos - 2] = kvInfo.hash;
  leaf->buffer_[writePos - 1] = kvInfo.vptr;

  leaf->ts.UpdateHeat();
  leaf->estimated_size_ += kvInfo.kvSize;
  leaf->buffer_size_ += kvInfo.kvSize;

  if (writePos < 32) {
    leaf->opt_lock_.WriteUnlock(true);
    return;
  }

  std::lock_guard<std::mutex> lk(leaf->flush_mutex_);

  // Group size_ and total size_ is updated when doing flush
  GetCompactor().updateSize(leaf->buffer_size_);
  leaf->heat_group_->updateSize(leaf->buffer_size_);
  leaf->buffer_size_ = 0;

  // Now we need to flush data into nvm
  SET_NODE_BUFFER_SIZE(leaf->status_, 0);
  assert(GET_NODE_BUFFER_SIZE(leaf->status_) == 0);

  auto nvmNode = leaf->nvm_node_;
  auto &nvmMeta = nvmNode->meta;
  int rows = GET_ROWS(nvmMeta.header);
  bool needSplit = rows == NVM_MAX_ROWS - 1;
  if (needSplit) {
    FlushBuffer(leaf, rows);
    level < 5 ? SplitLeafBelowLevel5(leaf) : SplitNode(leaf);
  } else {
    leaf->opt_lock_.WriteUnlock(true);
    FlushBuffer(leaf, rows);
  }
}

bool GlobalMemtable::FindKey(InnerNode *leaf, std::string &key, std::string &value) {
  std::string find_key;
  uint64_t vptr;
  uint64_t dummy_seq_num;

  uint64_t hash = Hash(key.c_str(), key.length(), kTypeValue);
  int pos = GET_NODE_BUFFER_SIZE(leaf->status_);
  auto buffer = leaf->buffer_;

  for (int i = 0; i < pos; ++i) {
    if (buffer[i * 2] != hash) {
      continue;
    }
    vlog_manager_->GetKeyValue(
        buffer[i * 2 + 1], find_key, value, dummy_seq_num);
    if (find_key == key) {
      return true;
    }
  }

  auto data = leaf->nvm_node_->data;
  auto fingerprints = leaf->nvm_node_->meta.fingerprints_;
  int rows = GET_ROWS(leaf->nvm_node_->meta.header);

  int search_rows = (rows - 1) / 2 + 1;
  int size = rows * 16;

  __m256i target = _mm256_set1_epi8((uint8_t)hash);
  for (int i = search_rows; i >= 0; --i) {
    int base = i << 5;
    __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(fingerprints + base));
    __m256i r = _mm256_cmpeq_epi8(f, target);
    auto res = (unsigned int)_mm256_movemask_epi8(r);
    while (res > 0) {
      int found = 31 - __builtin_clz(res);
      res -= (1 << found);
      int index = found + base;
      vptr = data[index * 2 + 1];
      if (index >= size || data[index * 2] != hash) {
        continue;
      }
      vlog_manager_->GetKeyValue(vptr, find_key, value, dummy_seq_num);
      if (key == find_key) {
        return true;
      }
    }
  }

  return false;
}

} // namespace ROCKSDB_NAMESPACE