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
#include "heat_group_manager.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

InnerNode::InnerNode()
    : heat_group_(nullptr), art(nullptr), artBackup(nullptr),
      nvm_node_(nullptr), backup_nvm_node_(nullptr), last_child_node_(nullptr),
      next_node_(nullptr), vptr_(0), status_(0), estimated_size_(0),
      buffer_size_(0) {
  memset(hll_, 0, 64);
  memset(buffer_, 0, 256);
}

void FlushBuffer(InnerNode* leaf, int row) {
  NVMNode* node = leaf->nvm_node_;
  uint64_t* buffer = leaf->buffer_;
  uint8_t* hll = leaf->hll_;

  int h, bucket;
  uint8_t digit;
  uint8_t fingerprints[16];
  for (size_t i = 0; i < 16; ++i) {
    auto hash = buffer[i << 1];
    fingerprints[i] = static_cast<uint8_t>(hash);

    bucket = (int)(hash & 63);
    h = (int)(hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    hll[bucket] = std::max(hll[bucket], digit);
  }

  memcpy(node->data + (row << 5), buffer, 256);
  memcpy(node->meta.fingerprints_ + (row << 4), fingerprints, 16);
  // _mm_clwb(data); _mm_sfence();

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, size + 16);
  SET_ROWS(hdr, row + 1);
  node->meta.header = hdr;
  // _mm_clwb(node->meta.header); _mm_sfence();
}

void FlushBufferAndRelease(InnerNode* leaf, int row) {
  NVMNode* node = leaf->nvm_node_;
  uint8_t* hll = leaf->hll_;

  uint64_t buffer[32];
  memcpy(buffer, leaf->buffer_, 256);

  leaf->opt_lock_.WriteUnlock(true);

  int h, bucket;
  uint8_t digit;
  uint8_t fingerprints[16];
  for (size_t i = 0; i < 16; ++i) {
    auto hash = buffer[i << 1];
    fingerprints[i] = static_cast<uint8_t>(hash);

    bucket = (int)(hash & 63);
    h = (int)(hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    hll[bucket] = std::max(hll[bucket], digit);
  }

  memcpy(node->data + (row << 5), buffer, 256);
  memcpy(node->meta.fingerprints_ + (row << 4), fingerprints, 16);
  // _mm_clwb(data); _mm_sfence();

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, size + 16);
  SET_ROWS(hdr, row + 1);
  node->meta.header = hdr;
  // _mm_clwb(node->meta.header); _mm_sfence();
}

//////////////////////////////////////////////////////////

GlobalMemtable::GlobalMemtable(
    VLogManager* vlog_manager, HeatGroupManager* group_manager, Env* env)
    : root_(nullptr), head_(nullptr), tail_(nullptr),
      vlog_manager_(vlog_manager), group_manager_(group_manager),
      env_(env) {
  vlog_manager->SetMemtable(this);
  InitFirstTwoLevel();
};

void GlobalMemtable::InitFirstTwoLevel() {
  root_ = new InnerNode();
  tail_ = new InnerNode();
  SET_NON_LEAF(root_->status_);
  SET_ART_FULL(root_->status_);

  // First level
  root_->art = AllocateArtNode(kNode256);
  auto art256 = (ArtNode256 *)root_->art;
  art256->header_.prefix_length_ = 1;
  art256->header_.num_children_ = 129;

  std::string ascii(
      "\t\n !\"#$%&'()*+,-./0123456789:;<=>?@"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~");
  std::unordered_set<int> used_ascii;
  used_ascii.insert(0);
  used_ascii.insert(LAST_CHAR);
  for (auto& c : ascii) {
    used_ascii.insert((int)c);
  }

  auto pre_allocate_children = used_ascii.size();

  InnerNode* next_inner_node = tail_;
  for (int first_char = LAST_CHAR; first_char >= 0; --first_char) {
    if (first_char > 128 && first_char != LAST_CHAR) {
      continue;
    }

    auto node = new InnerNode();
    SET_NON_LEAF(node->status_);
    SET_ART_FULL(node->status_);
    art256->children_[first_char] = node;

    // Second level
    node->art = AllocateArtNode(kNode256);
    auto art256_l2 = (ArtNode256 *)node->art;
    art256_l2->header_.prefix_length_ = 2;
    art256_l2->header_.num_children_ = pre_allocate_children;

    auto heat_group = new HeatGroup();

    for (int second_char = LAST_CHAR; second_char >= 0; --second_char) {
      if (used_ascii.find(second_char) == used_ascii.end()) {
        continue;
      }

      art256_l2->children_[second_char] = AllocateLeafNode(
          2, static_cast<unsigned char>(second_char), next_inner_node);
      art256_l2->children_[second_char]->heat_group_ = heat_group;
      next_inner_node = art256_l2->children_[second_char];
    }

    heat_group->group_manager_ = group_manager_;
    heat_group->first_node_ = art256_l2->children_[0];
    SET_GROUP_START(heat_group->first_node_->status_);
    heat_group->last_node_ = art256_l2->children_[LAST_CHAR];
    group_manager_->InsertIntoLayer(heat_group, BASE_LAYER);
  }

  head_ = next_inner_node;
}

void GlobalMemtable::Put(Slice& slice, uint64_t base_vptr, size_t count) {
  uint64_t vptr = base_vptr;
  slice.remove_prefix(WriteBatchInternal::kHeader);
  for (size_t c = 0; c < count; ++c) {
    auto record_start = slice.data();
    Slice key, value;
    ValueType type = ((ValueType *)slice.data())[0];
    slice.remove_prefix(RecordPrefixSize);
    GetLengthPrefixedSlice(&slice, &key);
    if (type == kTypeValue) {
      GetLengthPrefixedSlice(&slice, &value);
    }

    auto hash = Hash(key.data(), key.size());
    KVStruct kv_info(hash, vptr);
    kv_info.kv_size_ =
        type == kTypeDeletion
            ? 0 : (slice.data() - record_start) - RecordPrefixSize;
    Put(key, kv_info);
    vptr += (slice.data() - record_start);
  }
}

void GlobalMemtable::Put(Slice& key, KVStruct& kv_info) {
  size_t max_level = key.size();

  bool need_restart;
  size_t level = 1;
  uint32_t version;
  InnerNode* current = FindChild(root_, key[0]);
  InnerNode* next_node;

  while (true) {
  Restart:
    need_restart = false;
    version = current->opt_lock_.AwaitNodeUnlocked();

    if (IS_LEAF(current->status_)) {
      current->opt_lock_.UpgradeToWriteLockOrRestart(version, need_restart);
      if (need_restart) {
        goto Restart;
      }

      InsertIntoLeaf(current, kv_info, level);
      return;
    }

    // level > maxDepth means key need to be stored in vptr_
    if (level == max_level) {
      current->opt_lock_.UpgradeToWriteLock();
      current->vptr_ = kv_info.vptr_;
      current->opt_lock_.WriteUnlock(true);
      return;
    }

    next_node = FindChild(current, key[level]);
    current->opt_lock_.CheckOrRestart(version, need_restart);
    if (need_restart) {
      goto Restart;
    }

    if (!next_node) {
      current->opt_lock_.UpgradeToWriteLockOrRestart(version, need_restart);
      if (need_restart) {
        goto Restart;
      }

      InnerNode* leaf = AllocateLeafNode(level + 1, key[level], nullptr);
      ++leaf->status_;
      leaf->buffer_size_ += kv_info.kv_size_;
      leaf->buffer_[0] = kv_info.hash_;
      leaf->buffer_[1] = kv_info.vptr_;

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
      leaf->heat_group_->UpdateHeat();
      return;
    }

    current = next_node;
    ++level;
  }
}

bool GlobalMemtable::Get(std::string& key, std::string& value, Status* s) {
  size_t max_level = key.length();
  size_t level = 1;
  InnerNode* current = FindChild(root_, key[0]);

  while (true) {
    if (IS_LEAF(current->status_)) {
      return FindKeyInInnerNode(current, key, value, s);
    }

    if (level == max_level) {
      if (current->vptr_ == 0) {
        return false;
      }

      auto type = vlog_manager_->GetKeyValue(current->vptr_, key, value);
      *s = type == kTypeValue ? Status::OK() : Status::NotFound();
      return true;
    }

    current = FindChild(current, key[level++]);
    if (!current) {
      return false;
    }
  }
}

void GlobalMemtable::SqueezeNode(InnerNode* leaf) {
  auto node = leaf->nvm_node_;
  uint64_t* data = node->data;
  uint8_t* fingerprints = node->meta.fingerprints_;
  memset(leaf->hll_, 0, 64);

  // Maybe we can use vector instead of map.
  std::unordered_map<std::string, KVStruct> map;

  for (size_t i = 0; i < NVM_MAX_SIZE; i++) {
    uint64_t hash = data[(i << 1)];
    uint64_t vptr = data[(i << 1) + 1];
    if (!vptr) {
      continue;
    }
    std::string key;
    vlog_manager_->GetKey(vptr, key);
    map[key] = KVStruct(hash, vptr);
  }

  int count = 0, fpos = 0;
  auto prev_size = leaf->estimated_size_;
  int32_t cur_size = 0;
  memset(fingerprints, 0, 224);
  for (auto &pair: map) {
    auto & kv_info = pair.second;
    fingerprints[fpos++] = static_cast<uint8_t>(kv_info.hash_);
    data[count++] = kv_info.hash_;
    data[count++] = kv_info.vptr_;
    cur_size += kv_info.kv_size_;

    int bucket = kv_info.hash_ & 63;
    int h = (int) (kv_info.hash_ >> 6);
    uint8_t digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    leaf->hll_[bucket] = std::max(leaf->hll_[bucket], digit);
  }

  assert(fpos <= NVM_MAX_SIZE);
  assert(count <= (NVM_MAX_SIZE * 2));

  leaf->estimated_size_ = cur_size;
  leaf->heat_group_->UpdateSize(cur_size - prev_size);

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

void GlobalMemtable::SplitLeafBelowLevel5(InnerNode* leaf) {
  auto node = leaf->nvm_node_;
  auto* data = node->data;
  autovector<KVStruct> split_buckets[256];

  int level = GET_PRELEN(node->meta.header);
  for (size_t i = 0; i < NVM_MAX_SIZE; ++i) {
    KVStruct kv_info(data[(i << 1)], data[(i << 1) + 1]);
    if (!kv_info.vptr_) {
      continue;
    }
    split_buckets[static_cast<unsigned int>(GetPrefix(level, kv_info))]
        .push_back(kv_info); // ??
  }

  int64_t oldest_key_time = leaf->oldest_key_time_;

  // 0, 1 and 255
  int32_t split_num = 3;
  for (size_t i = 2; i < LAST_CHAR; ++i) {
    if (!split_buckets[i].empty()) {
      ++split_num;
    }
  }

  auto final_node = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), nullptr);
  auto last_node = final_node;
  auto first_node = final_node;
  final_node->oldest_key_time_ = oldest_key_time;

  std::vector<InnerNode*> new_leaves{final_node};
  std::vector<unsigned char> prefixes{LAST_CHAR};
  new_leaves.reserve(split_num);
  prefixes.reserve(split_num);

  for (int c = LAST_CHAR - 1; c > 0; --c) {
    if (split_buckets[c].empty() && c != 1) {
      continue;
    }
    auto new_leaf = AllocateLeafNode(
        level + 1, static_cast<unsigned char>(c), last_node);
    new_leaf->oldest_key_time_ = oldest_key_time;
    auto nvm_node = new_leaf->nvm_node_;
    int pos = 0, fpos = 0;
    int32_t leaf_size = 0;
    memset(nvm_node->meta.fingerprints_, 0, 224);
    for (auto& kv_info : split_buckets[c]) {
      nvm_node->meta.fingerprints_[fpos++] = static_cast<uint8_t>(kv_info.hash_);
      nvm_node->data[pos++] = kv_info.hash_;
      nvm_node->data[pos++] = kv_info.vptr_;
      leaf_size += kv_info.kv_size_;
    }
    new_leaf->estimated_size_ = leaf_size;
    assert(fpos <= NVM_MAX_SIZE);
    assert(pos <= (NVM_MAX_SIZE << 1));

    last_node = new_leaf;
    first_node = new_leaf;

    int rows = fpos > 0 ? ((fpos - 1) >> 4) + 1 : 0;
    int size = rows << 4;
    memset(nvm_node->data + pos,
           0, (size - fpos) << 4);

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
  leaf->vptr_ = split_buckets[0].empty() ? 0 : split_buckets[0].back().vptr_;
  SET_ART_NON_FULL(leaf->status_);

  InsertSplitInnerNode(leaf, first_node, final_node, level + 1);
  // _mm_sfence(); clwb(node->meta); _mm_sfence();
  leaf->estimated_size_ = 0;
  InsertNodesToGroup(leaf, new_leaves);

  SET_NON_LEAF(leaf->status_);
}

void GlobalMemtable::SplitLeaf(InnerNode* leaf) {
  auto node = leaf->nvm_node_;
  auto data = node->data;
  autovector<KVStruct> split_buckets[256];

  // TODO: remove duplication
  int level = GET_PRELEN(node->meta.header);
  uint8_t p = GET_LAST_PREFIX(node->meta.header);
  for (int i = NVM_MAX_SIZE - 1; i >= 0; --i) {
    uint64_t hash = data[(i << 1)];
    uint64_t vptr = data[(i << 1) + 1];
    if (!vptr) {
      continue;
    }
    std::string key;
    vlog_manager_->GetKey(vptr, key);
    split_buckets[static_cast<unsigned char>(key[level])].emplace_back(hash, vptr);
  }

  int64_t oldest_key_time = leaf->oldest_key_time_;

  // 0, 1 and 255 are always created
  int32_t split_num = 3;
  for (size_t i = 2; i < 255; ++i) {
    if (!split_buckets[i].empty()) {
      ++split_num;
    }
  }

  auto final_node = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), nullptr);
  auto last_node = final_node;
  auto first_node = final_node;
  final_node->oldest_key_time_ = oldest_key_time;

  std::vector<InnerNode*> new_leaves{final_node};
  std::vector<unsigned char> prefixes{LAST_CHAR};
  new_leaves.reserve(split_num);
  prefixes.reserve(split_num);

  for (int c = LAST_CHAR - 1; c > 0; --c) {
    if (split_buckets[c].empty() && c != 1) {
      continue;
    }
    auto new_leaf = AllocateLeafNode(
        level + 1, static_cast<unsigned char>(c), last_node);
    new_leaf->oldest_key_time_ = oldest_key_time;
    auto nvm_node = new_leaf->nvm_node_;
    int pos = 0, fpos = 0;
    int32_t leaf_size = 0;
    memset(nvm_node->meta.fingerprints_, 0, 224);
    for (auto& kv_info : split_buckets[c]) {
      nvm_node->meta.fingerprints_[fpos++] = static_cast<uint8_t>(kv_info.hash_);
      nvm_node->data[pos++] = kv_info.hash_;
      nvm_node->data[pos++] = kv_info.vptr_;
      leaf_size += kv_info.kv_size_;
    }
    new_leaf->estimated_size_ = leaf_size;

    last_node = new_leaf;
    first_node = new_leaf;

    int rows = fpos > 0 ? ((fpos - 1) >> 4) + 1 : 0;
    int size = rows << 4;
    memset(nvm_node->data + pos,
           0, (size - fpos) << 4);

    auto hdr = nvm_node->meta.header;
    SET_SIZE(hdr, size);
    SET_ROWS(hdr, rows);
    SET_LAST_PREFIX(hdr, c);
    SET_PRELEN(hdr, level + 1);
    nvm_node->meta.header = hdr;
    // _mm_clwb(nvm_node_->meta); _mm_clwb(hdr);

    new_leaves.push_back(new_leaf);
    prefixes.push_back(static_cast<unsigned char>(c));
  }

  std::reverse(new_leaves.begin(), new_leaves.end());
  std::reverse(prefixes.begin(), prefixes.end());
  auto art = AllocateArtAfterSplit(new_leaves, prefixes, leaf);
  art->prefix_length_ = level + 1;

  leaf->art = art;
  leaf->vptr_ = split_buckets[0].empty() ? 0 : split_buckets[0].back().vptr_;
  SET_ART_NON_FULL(leaf->status_);

  InsertSplitInnerNode(leaf, first_node, final_node, level + 1);
  // _mm_sfence(); clwb(node->meta); _mm_sfence();
  leaf->estimated_size_ = 0;
  InsertNodesToGroup(leaf, new_leaves);

  SET_NON_LEAF(leaf->status_);
}

// "4b prefix + 4b hash + 8b pointer" OR "4b SeqNum + 4b hash + 8b pointer"
// This function is responsible for unlocking opt_lock_
void GlobalMemtable::InsertIntoLeaf(
    InnerNode* leaf, KVStruct& kv_info, size_t level) {
  int writePos = GET_NODE_BUFFER_SIZE(++leaf->status_) << 1;
  assert(writePos <= 32);

  leaf->buffer_[writePos - 2] = kv_info.hash_;
  leaf->buffer_[writePos - 1] = kv_info.vptr_;
  leaf->buffer_size_ += kv_info.kv_size_;

  if (likely(writePos < 32)) {
    leaf->opt_lock_.WriteUnlock(true);
    leaf->heat_group_->UpdateHeat();
    return;
  }

  leaf->heat_group_->UpdateHeat();

  std::lock_guard<std::mutex> flush_lk(leaf->flush_mutex_);

  // Group size_ and total size_ is updated when doing flush
  auto buffer_size = leaf->buffer_size_;
  leaf->heat_group_->UpdateSize(buffer_size);
  leaf->estimated_size_ += buffer_size;
  leaf->buffer_size_ = 0;
  env_->GetCurrentTime(&leaf->oldest_key_time_);

  // Now we need to flush data into nvm
  SET_NODE_BUFFER_SIZE(leaf->status_, 0);
  assert(GET_NODE_BUFFER_SIZE(leaf->status_) == 0);

  auto nvmNode = leaf->nvm_node_;
  auto& nvmMeta = nvmNode->meta;
  int rows = GET_ROWS(nvmMeta.header);
  bool need_split = rows == NVM_MAX_ROWS - 1;

  if (likely(!need_split)) {
    // We cannot release the lock directly,
    // because we will use data in buffer
    // Two solutions:
    // 1. release opt lock after flushing data
    // 2. copy buffer and release lock, then flush (currently used)
    FlushBufferAndRelease(leaf, rows);
  } else {
    if (EstimateDistinctCount(leaf->hll_) < squeeze_threshold_) {
      FlushBufferAndRelease(leaf, rows);
      //std::lock_guard<std::mutex> gc_lk(leaf->gc_mutex_);
      SqueezeNode(leaf);
      return;
    } else {
      //std::lock_guard<std::mutex> gc_lk(leaf->gc_mutex_);
      FlushBuffer(leaf, rows);
      level < 5 ? SplitLeafBelowLevel5(leaf) : SplitLeaf(leaf);
      leaf->opt_lock_.WriteUnlock(false);
    }
  }
}

bool GlobalMemtable::FindKeyInInnerNode(
    InnerNode* leaf, std::string& key, std::string& value, Status* s) {
  std::string find_key;
  uint64_t vptr;
  ValueType type;

  uint64_t hash = Hash(key.c_str(), key.length(), kTypeValue);
  int pos = GET_NODE_BUFFER_SIZE(leaf->status_);
  auto buffer = leaf->buffer_;

  for (int i = 0; i < pos; ++i) {
    if (buffer[i * 2] != hash) {
      continue;
    }
    type = vlog_manager_->GetKeyValue(
        buffer[i * 2 + 1], find_key, value);
    if (find_key == key) {
      *s = type == kTypeValue ? Status::OK() : Status::NotFound();
      return true;
    }
  }

  __m256i target = _mm256_set1_epi8((uint8_t)hash);

  std::vector<NVMNode*> nvm_nodes{leaf->nvm_node_};
  auto backup_nvm_node = leaf->backup_nvm_node_;
  if (backup_nvm_node && backup_nvm_node != nvm_nodes[0]) {
    nvm_nodes.push_back(backup_nvm_node);
  }

  for (auto& nvm_node : nvm_nodes) {
    auto data = nvm_node->data;
    auto fingerprints = nvm_node->meta.fingerprints_;
    int rows = GET_ROWS(nvm_node->meta.header);

    int search_rows = (rows + 1) / 2 - 1;
    int size = rows * 16;

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
        type = vlog_manager_->GetKeyValue(vptr, find_key, value);
        if (key == find_key) {
          *s = type == kTypeValue ? Status::OK() : Status::NotFound();
          return true;
        }
      }
    }
  }

  return false;
}

InnerNode* GlobalMemtable::FindInnerNodeByKey(
    std::string& key, size_t& level, bool& stored_in_nvm) {
  size_t max_level = key.size();

  level = 1;
  stored_in_nvm = true;
  InnerNode* current = FindChild(root_, key[0]);

  while (true) {
    if (IS_LEAF(current->status_)) {
      return current;
    }

    if (level == max_level) {
      stored_in_nvm = false;
      return current;
    }

    current = FindChild(current, key[level++]);
    if (!current) {
      return nullptr;
    }
  }
}

} // namespace ROCKSDB_NAMESPACE