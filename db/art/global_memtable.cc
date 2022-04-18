//
// Created by joechen on 2022/4/3.
//

#include "global_memtable.h"

#include <cassert>
#include <unordered_map>

#include <db/write_batch_internal.h>

#include "utils.h"
#include "nvm_node.h"
#include "art_node.h"
#include "vlog_manager.h"
#include "heat_group.h"
#include "heat_group_manager.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

#define SQUEEZE_THRESHOLD 112

InnerNode::InnerNode()
    : heat_group_(nullptr), art(nullptr),
      nvm_node_(nullptr), backup_nvm_node_(nullptr),
      last_child_node_(nullptr), next_node_(nullptr),
      vptr_(0), estimated_size_(0), buffer_size_(0),
      status_(0), oldest_key_time_(0) {
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

  auto data_flush_start = node->data + (row << 5);
  auto finger_flush_start = node->meta.fingerprints_ + ROW_TO_SIZE(row);
  MEMCPY(data_flush_start, buffer, ROW_SIZE, PMEM_F_MEM_NODRAIN);
  MEMCPY(finger_flush_start, fingerprints, 16, PMEM_F_MEM_NODRAIN);

  MEMORY_BARRIER;

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, size + 16);
  SET_ROWS(hdr, row + 1);
  node->meta.header = hdr;
  PERSIST(node, 8);
}

#ifdef ROCKSDB_SUPPORT_THREAD_LOCAL
thread_local uint8_t fingerprints_copy[16];
thread_local uint64_t buffer_copy[32];
#endif

void FlushBufferAndRelease(InnerNode* leaf, int row) {
  NVMNode* node = leaf->nvm_node_;
  uint8_t* hll = leaf->hll_;

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t fingerprints_copy[16];
  uint64_t buffer_copy[32];
#endif

  memcpy(buffer_copy, leaf->buffer_, ROW_SIZE);

  leaf->opt_lock_.WriteUnlock(true);

  int h, bucket;
  uint8_t digit;
  for (size_t i = 0; i < 16; ++i) {
    auto hash = buffer_copy[i << 1];
    fingerprints_copy[i] = static_cast<uint8_t>(hash);
    bucket = (int)(hash & 63);
    h = (int)(hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    hll[bucket] = std::max(hll[bucket], digit);
  }

  uint64_t* data_flush_start = node->data + (row << 5);
  uint8_t* finger_flush_start = node->meta.fingerprints_ + ROW_TO_SIZE(row);
  MEMCPY(data_flush_start, buffer_copy, 256,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);
  MEMCPY(finger_flush_start, fingerprints_copy, 16,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

  MEMORY_BARRIER;

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, size + 16);
  SET_ROWS(hdr, row + 1);
  node->meta.header = hdr;
  PERSIST(node, 8);
}

//////////////////////////////////////////////////////////

GlobalMemtable::GlobalMemtable(
    VLogManager* vlog_manager, HeatGroupManager* group_manager, Env* env)
    : root_(nullptr), head_(nullptr), tail_(nullptr),
      vlog_manager_(vlog_manager), group_manager_(group_manager),
      env_(env) {
  vlog_manager->SetMemtable(this);
  InitFirstLevel();
}

void GlobalMemtable::InitFirstLevel() {
  root_ = new InnerNode();
  tail_ = new InnerNode();
  SET_NON_LEAF(root_->status_);
  SET_ART_FULL(root_->status_);

  // First level
  root_->art = AllocateArtNode(kNode256);
  auto art256 = (ArtNode256*)root_->art;
  art256->header_.num_children_ = 256;
  art256->header_.art_type_ = kNode256;

  /*std::string ascii(
      "\t\n !\"#$%&'()*+,-./0123456789:;<=>?@"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~");
  std::unordered_set<int> used_ascii;
  used_ascii.insert(0);
  used_ascii.insert(LAST_CHAR);
  for (auto& c : ascii) {
    used_ascii.insert((int)c);
  }

  auto pre_allocate_children = used_ascii.size();*/

  InnerNode* next_inner_node = tail_;
  /* for (int first_char = LAST_CHAR; first_char >= 0; --first_char) {
    auto inner_node = AllocateLeafNode(
        1, static_cast<unsigned char>(first_char), next_inner_node);
    auto group_start_node = AllocateLeafNode(
        0, 0, inner_node);

    auto heat_group = new HeatGroup();
    SET_GROUP_START(group_start_node->status_);
    heat_group->first_node_ = group_start_node;
    heat_group->last_node_ = inner_node;
    heat_group->group_manager_ = group_manager_;
    inner_node->heat_group_ = heat_group;
    group_start_node->heat_group_ = heat_group;
    group_manager_->InsertIntoLayer(heat_group, BASE_LAYER);

    art256->children_[first_char] = inner_node;
    next_inner_node = group_start_node;

    FLUSH(inner_node->nvm_node_, CACHE_LINE_SIZE);
  } */

  for (int first_char = LAST_CHAR; first_char >= 0;) {
    auto heat_group = new HeatGroup();
    for (int n = 0; n < 4; ++n, --first_char) {
      auto inner_node = AllocateLeafNode(
          1, static_cast<unsigned char>(first_char), next_inner_node);
      art256->children_[first_char] = inner_node;
      inner_node->heat_group_ = heat_group;
      next_inner_node = inner_node;
      FLUSH(inner_node->nvm_node_, CACHE_LINE_SIZE);
    }

    auto group_start_node = AllocateLeafNode(
        0, 0, next_inner_node);
    FLUSH(group_start_node->nvm_node_, CACHE_LINE_SIZE);

    SET_GROUP_START(group_start_node->status_);
    heat_group->first_node_ = group_start_node;
    heat_group->last_node_ = art256->children_[first_char + 4];
    heat_group->group_manager_ = group_manager_;
    group_start_node->heat_group_ = heat_group;
    group_manager_->InsertIntoLayer(heat_group, BASE_LAYER);

    next_inner_node = group_start_node;
  }

  MEMORY_BARRIER;

  head_ = next_inner_node;
}

void GlobalMemtable::Put(Slice& slice, uint64_t base_vptr, size_t count) {
  uint64_t vptr = base_vptr;
  uint32_t val_len = 0;
  Slice key;

  slice.remove_prefix(WriteBatchInternal::kHeader);
  for (size_t c = 0; c < count; ++c) {
    auto record_start = slice.data();
    ValueType type = ((ValueType*)slice.data())[0];
    slice.remove_prefix(WriteBatchInternal::kRecordPrefixSize);
    GetLengthPrefixedSlice(&slice, &key);
    if (type == kTypeValue) {
      GetVarint32(&slice, &val_len);
      slice.remove_prefix(val_len);
    }

    auto hash = HashOnly(key.data(), key.size());
    KVStruct kv_info(hash, vptr);
    kv_info.kv_size_ =
        type == kTypeDeletion
            ? 0 : (slice.data() - record_start) -
                  WriteBatchInternal::kRecordPrefixSize;
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

      Rehash(key.data(), key.size(), kv_info.hash_, level);
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

      Rehash(key.data(), key.size(), kv_info.hash_, level + 1);
      InnerNode* leaf = AllocateLeafNode(level + 1, key[level], nullptr);
      ++leaf->status_;
      leaf->buffer_size_ += kv_info.kv_size_;
      leaf->buffer_[0] = kv_info.hash_;
      leaf->buffer_[1] = kv_info.vptr_;

      if (IS_ART_FULL(current->status_)) {
        current->art = ReallocateArtNode(current->art);
        SET_ART_NON_FULL(current->status_);
      }

      InsertToArtNode(current, leaf, key[level], true);
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
      return FindKeyInInnerNode(current, level, key, value, s);
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
  auto data = node->data;
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

  int32_t prev_size = leaf->estimated_size_;
  int32_t cur_size = 0;

  int count = 0, fpos = 0;
  uint8_t temp_fingerprints[224] = {0};
  uint64_t temp_data[448] = {0};

  for (auto& pair: map) {
    auto& kv_info = pair.second;
    temp_fingerprints[fpos++] = static_cast<uint8_t>(kv_info.hash_);
    temp_data[count++] = kv_info.hash_;
    temp_data[count++] = kv_info.vptr_;
    cur_size += kv_info.kv_size_;

    int bucket = (int)(kv_info.hash_ & 63);
    int h = (int) (kv_info.hash_ >> 6);
    uint8_t digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    leaf->hll_[bucket] = std::max(leaf->hll_[bucket], digit);
  }

  assert(fpos <= NVM_MAX_SIZE);
  assert(count <= (NVM_MAX_SIZE * 2));

  leaf->estimated_size_ = cur_size;
  leaf->heat_group_->UpdateSize(cur_size - prev_size);

  int flush_size = ALIGN_UP(fpos, 16);
  int flush_rows = SIZE_TO_ROWS(flush_size);
  MEMCPY(node->data, temp_data,
         SIZE_TO_BYTES(flush_size), PMEM_F_MEM_NODRAIN);
  MEMCPY(node->meta.fingerprints_,
         temp_fingerprints, flush_size, PMEM_F_MEM_NODRAIN);

  MEMORY_BARRIER;

  uint64_t hdr = node->meta.header;
  SET_SIZE(hdr, flush_size);
  SET_ROWS(hdr, flush_rows);
  node->meta.header = hdr;
  PERSIST(node, 8);
}

// For level 1, 4, 7..., we read key and modify prefixes in hash.
void GlobalMemtable::ReadFromVLog(NVMNode* nvm_node, size_t level,
                                  uint64_t& leaf_vptr,
                                  autovector<KVStruct>* split_buckets) {
  auto data = nvm_node->data;
  for (int i = NVM_MAX_SIZE - 1; i >= 0; --i) {
    KVStruct kv_info(data[(i << 1)], data[(i << 1) + 1]);
    if (!kv_info.vptr_) {
      continue;
    }

    std::string key;
    vlog_manager_->GetKey(kv_info.vptr_, key);

    if (key.length() == level) {
      leaf_vptr = kv_info.vptr_;
      continue;
    }

    Rehash(key.c_str(), key.size(), kv_info.hash_, level + 1);
    split_buckets[static_cast<unsigned char>(key[level])]
        .push_back(kv_info);
  }
}

void GlobalMemtable::ReadFromNVM(NVMNode* nvm_node, size_t level,
                                 uint64_t& leaf_vptr,
                                 autovector<KVStruct>* split_buckets) {
  auto data = nvm_node->data;
  for (size_t i = 0; i < NVM_MAX_SIZE; ++i) {
    KVStruct kv_info(data[(i << 1)], data[(i << 1) + 1]);
    if (!kv_info.vptr_) {
      continue;
    }

    if (kv_info.key_length_ == level) {
      leaf_vptr = kv_info.vptr_;
    } else {
      split_buckets[static_cast<unsigned char>(GetPrefix(kv_info, level))]
          .push_back(kv_info);
    }
  }
}

void GlobalMemtable::SplitLeaf(InnerNode* leaf, size_t level,
                               InnerNode** node_need_split) {

  auto old_nvm_node = leaf->nvm_node_;
  auto old_leaf_data = old_nvm_node->data;
  *node_need_split = nullptr;

  uint64_t leaf_vptr = 0;
  autovector<KVStruct> split_buckets[256];
  if (level % 3 == 0) {
    ReadFromVLog(old_nvm_node, level, leaf_vptr, split_buckets);
  } else {
    ReadFromNVM(old_nvm_node, level, leaf_vptr, split_buckets);
  }

  int64_t oldest_key_time = leaf->oldest_key_time_;

  // 0 and 255 are always created
  int32_t split_num = 1;
  for (size_t i = 1; i <= LAST_CHAR; ++i) {
    if (!split_buckets[i].empty()) {
      ++split_num;
    }
  }

  auto dummy_node = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), nullptr);
  dummy_node->oldest_key_time_ = oldest_key_time;
  SET_NON_LEAF(dummy_node->status_);

  auto last_node = dummy_node;
  auto first_node = dummy_node;
  auto final_node = dummy_node;

  std::vector<InnerNode*> new_leaves;
  std::vector<unsigned char> prefixes;
  new_leaves.reserve(split_num + 1);
  prefixes.reserve(split_num);

  uint8_t temp_fingerprints[224] = {0};
  uint64_t temp_data[448] = {0};

  for (int c = LAST_CHAR; c >= 0; --c) {
    if (split_buckets[c].empty() && c > 0) {
      continue;
    }
    auto new_leaf = AllocateLeafNode(
        level + 1, static_cast<unsigned char>(c), last_node);
    new_leaf->oldest_key_time_ = oldest_key_time;
    auto nvm_node = new_leaf->nvm_node_;
    int pos = 0, fpos = 0;
    int32_t leaf_size = 0;
    for (auto& kv_info : split_buckets[c]) {
      temp_fingerprints[fpos++] = static_cast<uint8_t>(kv_info.hash_);
      temp_data[pos++] = kv_info.hash_;
      temp_data[pos++] = kv_info.vptr_;
      leaf_size += kv_info.kv_size_;
    }
    new_leaf->estimated_size_ = leaf_size;

    last_node = new_leaf;
    first_node = new_leaf;

    int flush_size = ALIGN_UP(fpos, 16);
    int flush_rows = SIZE_TO_ROWS(flush_size);
    assert(flush_rows <= NVM_MAX_ROWS);
    assert(flush_size <= NVM_MAX_SIZE);

    memset(temp_data + pos, 0, SIZE_TO_BYTES(flush_size - fpos));
    memset(temp_fingerprints + fpos, 0, flush_size - fpos);
    MEMCPY(nvm_node->data, temp_data,
           SIZE_TO_BYTES(flush_size), PMEM_F_MEM_NODRAIN);
    MEMCPY(nvm_node->meta.fingerprints_, temp_fingerprints,
           flush_size, PMEM_F_MEM_NODRAIN);

    auto hdr = nvm_node->meta.header;
    SET_SIZE(hdr, flush_size);
    SET_ROWS(hdr, flush_rows);
    SET_LAST_PREFIX(hdr, c);
    SET_LEVEL(hdr, level + 1);
    nvm_node->meta.header = hdr;
    FLUSH(nvm_node, 8);

    new_leaves.push_back(new_leaf);
    prefixes.push_back(static_cast<unsigned char>(c));

    if (flush_size >= 208) {
      *node_need_split = new_leaf;
    }
  }

  MEMORY_BARRIER;

  std::reverse(new_leaves.begin(), new_leaves.end());
  std::reverse(prefixes.begin(), prefixes.end());
  auto art = AllocateArtAfterSplit(new_leaves, prefixes);

  leaf->art = art;
  leaf->vptr_ = leaf_vptr;
  SET_ART_NON_FULL(leaf->status_);

  new_leaves.push_back(final_node);
  InsertSplitInnerNode(leaf, first_node, final_node, level + 1);
  leaf->estimated_size_ = 0;
  InsertNodesToGroup(leaf, new_leaves);

  SET_NON_LEAF(leaf->status_);
}

// "4b prefix + 4b hash + 8b pointer" OR "4b SeqNum + 4b hash + 8b pointer"
// This function is responsible for unlocking opt_lock_
void GlobalMemtable::InsertIntoLeaf(InnerNode* leaf, KVStruct& kv_info,
                                    size_t level) {
  int write_pos = GET_NODE_BUFFER_SIZE(++leaf->status_) << 1;
  assert(write_pos <= 32);

  leaf->buffer_[write_pos - 2] = kv_info.hash_;
  leaf->buffer_[write_pos - 1] = kv_info.vptr_;
  leaf->buffer_size_ += kv_info.kv_size_;

  if (likely(write_pos < 32)) {
    leaf->opt_lock_.WriteUnlock(true);
    leaf->heat_group_->UpdateHeat();
    return;
  }

  leaf->heat_group_->UpdateHeat();

  InnerNode* next_to_split = nullptr;

  {
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

    int rows = GET_ROWS(leaf->nvm_node_->meta.header);
    assert(rows < NVM_MAX_ROWS);

    if (likely(rows < NVM_MAX_ROWS - 1)) {
      // We can't release the lock directly,
      // because we will use data in buffer
      // Two solutions:
      // 1. release opt lock after flushing data
      // 2. copy buffer and release lock, then flush (currently used)
      // 3. use double buffer, but memory usage of the tree
      //    will be about 1.5 times higher.
      FlushBufferAndRelease(leaf, rows);
    } else {
      if (EstimateDistinctCount(leaf->hll_) < SQUEEZE_THRESHOLD) {
        FlushBufferAndRelease(leaf, rows);
        //std::lock_guard<std::mutex> gc_lk(leaf->gc_mutex_);
        SqueezeNode(leaf);
        return;
      } else {
        //std::lock_guard<std::mutex> gc_lk(leaf->gc_mutex_);
        FlushBuffer(leaf, rows);
        SplitLeaf(leaf, level, &next_to_split);

        if (next_to_split) {
          next_to_split->opt_lock_.UpgradeToWriteLock();
          next_to_split->flush_mutex_.lock();
        }

        leaf->opt_lock_.WriteUnlock(false);
      }
    }
  }

  while (next_to_split) {
    InnerNode* current = next_to_split;
    SplitLeaf(current, ++level, &next_to_split);

    if (next_to_split) {
      next_to_split->opt_lock_.UpgradeToWriteLock();
      next_to_split->flush_mutex_.lock();
    }
    current->opt_lock_.WriteUnlock();
    current->flush_mutex_.unlock();
  }
}

bool GlobalMemtable::FindKeyInInnerNode(InnerNode* leaf, size_t level,
                                        std::string& key, std::string& value,
                                        Status* s) {
  std::string find_key;
  uint64_t vptr;
  ValueType type;

  uint64_t hash = HashAndPrefix(key.c_str(), key.length(), level);
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

    for (int row = search_rows; row >= 0; --row) {
      int base = row << 5;
      __m256i f = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(fingerprints + base));
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

InnerNode* GlobalMemtable::FindInnerNodeByKey(std::string& key,
                                              size_t& level,
                                              bool& stored_in_nvm) {
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