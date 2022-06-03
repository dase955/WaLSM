//
// Created by joechen on 2022/4/3.
//

#include "global_memtable.h"

#include <cassert>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

#include <db/write_batch_internal.h>

#include "utils.h"
#include "nvm_node.h"
#include "art_node.h"
#include "vlog_manager.h"
#include "heat_group.h"
#include "heat_group_manager.h"
#include "node_allocator.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

#define SQUEEZE_THRESHOLD 104

InnerNode::InnerNode()
    : heat_group_(nullptr), art(nullptr), backup_art(nullptr),

      nvm_node_(nullptr), backup_nvm_node_(nullptr),
      support_node_(nullptr), next_node_(nullptr),
      vptr_(0), estimated_size_(0), squeezed_size_(0),
      status_(0), oldest_key_time_(0) {
  memset(hll_, 0, 64);
  memset(buffer_, 0, 256);
}

#ifdef ROCKSDB_SUPPORT_THREAD_LOCAL
thread_local uint8_t fingerprints_copy[16];
#endif

void FlushBuffer(InnerNode* leaf, int row) {
  NVMNode* node = leaf->nvm_node_;
  uint8_t* hll = leaf->hll_;

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t fingerprints_copy[16];
#endif

  int h, bucket;
  uint8_t digit;
  for (size_t i = 0; i < 16; ++i) {
    auto hash = leaf->buffer_[i << 1];
    fingerprints_copy[i] = static_cast<uint8_t>(hash);
    bucket = (int)(hash & 63);
    h = (int)(hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    hll[bucket] = std::max(hll[bucket], digit);
  }

  uint64_t* data_flush_start = node->data + (row << 5);
  uint8_t* finger_flush_start = node->meta.fingerprints_ + ROW_TO_SIZE(row);
  MEMCPY(data_flush_start, leaf->buffer_, 256,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);
  MEMCPY(finger_flush_start, fingerprints_copy, 16,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

  NVM_BARRIER;

  // Write metadata
  uint64_t hdr = node->meta.header;
  uint8_t size = GET_SIZE(hdr);
  SET_SIZE(hdr, size + 16);
  SET_ROWS(hdr, row + 1);
  node->meta.header = hdr;
  PERSIST(node, 8);

  SET_NODE_BUFFER_SIZE(leaf->status_, 0);
}

//////////////////////////////////////////////////////////

GlobalMemtable::GlobalMemtable(
    VLogManager* vlog_manager, HeatGroupManager* group_manager, Env* env,
    bool recovery)
    : root_(nullptr), vlog_manager_(vlog_manager),
      group_manager_(group_manager), env_(env) {
  vlog_manager->SetMemtable(this);
  recovery ? Recovery() : InitFirstLevel();
}

GlobalMemtable::~GlobalMemtable() {
  uint64_t inode_vptrs[262144] = {0};
  int count = 0;
  DeleteInnerNode(root_, inode_vptrs, count);

  int fd = open("inode_vptrs", O_CREAT | O_RDWR | O_DIRECT);
  write(fd, inode_vptrs, 2097152);
  close(fd);
}

void CheckHeatGroup(HeatGroup* group) {
  auto start = group->first_node_;
  auto end = group->last_node_;
  assert(IS_GROUP_START(start->status_));
  assert(start == end || !IS_GROUP_START(end->status_));
  assert(end->heat_group_ = group);
  assert(group->group_manager_);

  if (start == end) {
    return;
  }

  auto cur = start->next_node_;
  while (cur != end) {
    assert(cur->heat_group_ == group);
    assert(!IS_GROUP_START(cur->status_));
    cur = cur->next_node_;
  }
}


InnerNode* GlobalMemtable::RecoverNonLeaf(InnerNode* parent, int level,
                                          HeatGroup*& group) {
  NVMNode* cur = parent->nvm_node_;
  parent->vptr_ = cur->meta.node_info;
  parent->heat_group_ = group;
  SET_NON_LEAF(parent->status_);

  std::vector<InnerNode*> children;
  std::vector<unsigned char> prefixes;
  InnerNode* last_inner_node = parent;

  while (true) {
    cur = GetNextNode(cur);

    int cur_level = GET_LEVEL(cur->meta.header);
    if (cur_level > level) {
      SET_NON_LEAF(last_inner_node->status_);
      last_inner_node = RecoverNonLeaf(last_inner_node, cur_level, group);
      cur = GetNextNode(last_inner_node->nvm_node_);
    }

    auto inner_node = RecoverInnerNode(cur);
    inner_node->heat_group_ = group;
    inner_node->parent_node_ = parent;
    last_inner_node->next_node_ = inner_node;
    group->last_node_ = last_inner_node;
    group->group_size_.fetch_add(inner_node->estimated_size_);
    UpdateTotalSize(inner_node->estimated_size_);

    // Group Head
    if (GET_TAG(cur->meta.header, GROUP_START_TAG)) {
      group->group_manager_ = group_manager_;
      CheckHeatGroup(group);
      group_manager_->InsertIntoLayer(group, BASE_LAYER);
      group = new HeatGroup();
      group->first_node_ = inner_node;
      inner_node->parent_node_ = nullptr;
      inner_node->heat_group_ = group;
      SET_GROUP_START(inner_node->status_);
      SET_NON_LEAF(inner_node->status_);
      last_inner_node = inner_node;
      continue;
    }

    if (GET_TAG(cur->meta.header, DUMMY_TAG)) {
      assert((int)GET_LEVEL(cur->meta.header) == level);
      parent->art = AllocateArtAfterSplit(children, prefixes);
      parent->support_node_ = inner_node;
      return inner_node;
    }

    assert((int)GET_LEVEL(inner_node->nvm_node_->meta.header) == level);
    auto prefix = GET_LAST_PREFIX(cur->meta.header);
    children.push_back(inner_node);
    prefixes.push_back(prefix);
    last_inner_node = inner_node;
  }

  return nullptr;
}

void GlobalMemtable::Recovery() {
  root_ = RecoverInnerNode(GetNodeAllocator()->GetHead());
  SET_ART_FULL(root_->status_);
  SET_GROUP_START(root_->status_);

  HeatGroup* group = new HeatGroup;
  group->first_node_ = root_;

  [[maybe_unused]] auto tail = RecoverNonLeaf(root_, 1, group);

  group->group_manager_ = group_manager_;
  CheckHeatGroup(group);
  group_manager_->InsertIntoLayer(group, BASE_LAYER);

#ifndef NDEBUG
  auto cur = root_;
  while (cur->next_node_) {
    cur = cur->next_node_;
  }
  assert(cur == tail);
  assert((char*)(tail->nvm_node_) - (char*)(root_->nvm_node_) == PAGE_SIZE);
#endif

  // Recover vptr in leaf nodes
  uint64_t inode_vptrs[262144] = {0};
  int fd = open("inode_vptrs", O_CREAT | O_RDWR | O_DIRECT);
  read(fd, inode_vptrs, 2097152);
  close(fd);

  int count = 0;
  while (count < 262144 && inode_vptrs[count]) {
    PutRecover(inode_vptrs[count++]);
  }

}

void GlobalMemtable::InitFirstLevel() {
  root_ = AllocateLeafNode(0, 0, nullptr);
  auto tail = AllocateLeafNode(1, 0, nullptr);
  tail->parent_node_ = root_;
  SET_TAG(tail->nvm_node_->meta.header, DUMMY_TAG);
  FLUSH(tail->nvm_node_, CACHE_LINE_SIZE);
  SET_NON_LEAF(root_->status_);
  SET_ART_FULL(root_->status_);

  // First level
  root_->support_node_ = tail;
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

  InnerNode* next_inner_node = tail;
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
      inner_node->parent_node_ = root_;
      next_inner_node = inner_node;
      FLUSH(inner_node->nvm_node_, CACHE_LINE_SIZE);
    }

    auto group_start_node = AllocateLeafNode(
        0, 0, next_inner_node);
    SET_GROUP_START(group_start_node->status_);
    SET_NON_LEAF(group_start_node->status_);
    SET_TAG(group_start_node->nvm_node_->meta.header, GROUP_START_TAG);
    FLUSH(group_start_node->nvm_node_, CACHE_LINE_SIZE);

    heat_group->first_node_ = group_start_node;
    heat_group->last_node_ = art256->children_[first_char + 4];
    heat_group->group_manager_ = group_manager_;
    group_start_node->heat_group_ = heat_group;
    group_manager_->InsertIntoLayer(heat_group, BASE_LAYER);

    next_inner_node = group_start_node;
  }

  root_->next_node_ = next_inner_node;
  root_->nvm_node_->meta.next1 =
      GetNodeAllocator()->relative(next_inner_node->nvm_node_);

  FLUSH(root_->nvm_node_, CACHE_LINE_SIZE);

  NVM_BARRIER;
}

void GlobalMemtable::PutRecover(uint64_t vptr) {
  std::string key, value;
  auto type = vlog_manager_->GetKeyValue(vptr, key, value);

  size_t level = 0;
  size_t max_level = key.size();
  InnerNode* current = root_;

  while (true) {
    current = FindChild(current, key[++level]);
    if (level == max_level) {
      uint64_t hash = HashAndPrefix(key.c_str(), key.length(), level);
      current->hash_ = hash;
      current->vptr_ = vptr;
      return;
    }
  }
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
    kv_info.kv_size_ = slice.data() - record_start;
    Put(key, kv_info);
    vptr += (slice.data() - record_start);
  }
}

void GlobalMemtable::Put(Slice& key, KVStruct& kv_info) {
  size_t max_level = key.size();

  uint32_t version;
  InnerNode* first = FindChild(root_, key[0]);

Restart:
  size_t level = 1;
  InnerNode* current = first;
  InnerNode* next_node = nullptr;

  while (true) {
LocalRestart:
    version = current->opt_lock_.AwaitUnlocked();

    if (IS_LEAF(current->status_)) {
      if (!current->opt_lock_.TryLock(version)) {
        goto LocalRestart;
      }

      if (unlikely(IS_INVALID(current->status_))) {
        current->opt_lock_.unlock();
        goto Restart;
      }

      Rehash(key.data(), key.size(), kv_info.hash_, level);
      InsertIntoLeaf(current, kv_info, level);
      return;
    }

    if (level == max_level) {
      std::lock_guard<RWSpinLock> vptr_lk(current->vptr_lock_);
      if (unlikely(IS_LEAF(current->status_))) {
        goto LocalRestart;
      }

      Rehash(key.data(), key.size(), kv_info.hash_, level);
      current->hash_ = kv_info.hash_;
      current->vptr_ = kv_info.vptr_;
      return;
    }

    next_node = FindChild(current, key[level]);
    if (!next_node) {
      if (!current->opt_lock_.TryLock(version)) {
        goto LocalRestart;
      }

      if (IS_LEAF(current->status_)) {
        current->opt_lock_.unlock();
        goto LocalRestart;
      }

      Rehash(key.data(), key.size(), kv_info.hash_, level + 1);
      InnerNode* leaf = AllocateLeafNode(level + 1, key[level], nullptr);

      std::lock_guard<OptLock> leaf_lk(leaf->opt_lock_);

      ++leaf->status_;
      leaf->buffer_[0] = kv_info.hash_;
      leaf->buffer_[1] = kv_info.vptr_;
      leaf->estimated_size_ = kv_info.kv_size_;
      leaf->parent_node_ = current;

      InsertToArtNode(current, leaf, key[level], true);
      current->opt_lock_.unlock();
      leaf->heat_group_->UpdateSize(kv_info.kv_size_);
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
  InnerNode* backup_node = nullptr;
  size_t backup_level = 0;
  InnerNode* current = FindChild(root_, key[0]);

  // assume first level will not become non-leaf again
  bool found = false;
  while (current && !found) {
    if (IS_LEAF(current->status_)) {
      found = FindKeyInInnerNode(current, level, key, value, s);
    } else if (level == max_level) {
      shared_lock<RWSpinLock> read_lk(current->vptr_lock_);

      // This mean children of current node is waiting to be reclaimed,
      // so this node becomes a leaf node again.
      if (unlikely(IS_LEAF(current->status_))) {
        found = FindKeyInInnerNode(current, level, key, value, s);
      } else if (current->vptr_ > 0) {
        auto type = vlog_manager_->GetKeyValue(current->vptr_, key, value);
        *s = type == kTypeValue ? Status::OK() : Status::NotFound();
        found = true;
      }

      break;
    }

    current = FindChild(current, key, level++, &backup_node, backup_level);
  }

  if (!found && backup_node) {
    found = FindKeyInInnerNode(backup_node, backup_level, key, value, s);
  }

  if (backup_level) {
    ReduceBackupRead();
  }

  return found;
}

#ifdef ROCKSDB_SUPPORT_THREAD_LOCAL
thread_local uint8_t temp_fingerprints[224] = {0};
thread_local uint64_t temp_data[448] = {0};
#endif

void GlobalMemtable::SqueezeNode(InnerNode* leaf) {
  auto node = leaf->nvm_node_;
  auto data = node->data;

  // Maybe we can use vector instead of map.
  std::unordered_set<std::string> key_set;

  RecordIndex index;
  std::unordered_map<uint64_t, std::vector<RecordIndex>> unused_indexes;

  int32_t prev_size = leaf->estimated_size_;
  int32_t cur_size = 0;
  int count = 0, fpos = 0;

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t temp_fingerprints[224] = {0};
  uint64_t temp_data[448] = {0};
#endif

  for (int i = NVM_MAX_SIZE - 1; i >= 0; --i) {
    uint64_t hash = data[(i << 1)];
    uint64_t vptr = data[(i << 1) + 1];
    auto kv_info = KVStruct(hash, vptr);
    if (!vptr) {
      continue;
    }
    std::string key;
    vlog_manager_->GetKeyIndex(vptr, key, index);
    if (key_set.find(key) != key_set.end()) {
      GetActualVptr(vptr);
      unused_indexes[vptr >> 20].emplace_back(index);
      continue;
    }

    key_set.emplace(key);
    temp_fingerprints[fpos++] = static_cast<uint8_t>(kv_info.hash_);
    temp_data[count++] = kv_info.hash_;
    temp_data[count++] = kv_info.vptr_;
    cur_size += kv_info.kv_size_;
  }

  assert(fpos <= NVM_MAX_SIZE - 16);
  assert(fpos * 2 == count);

  leaf->estimated_size_ = cur_size;
  leaf->squeezed_size_ += (prev_size - cur_size);
  leaf->heat_group_->UpdateSqueezedSize(prev_size - cur_size);

  int flush_size = ALIGN_UP(fpos, 16);
  int flush_rows = SIZE_TO_ROWS(flush_size);
  assert(flush_rows <= NVM_MAX_ROWS);
  assert(flush_size <= NVM_MAX_SIZE);

  memset(temp_data + count, 0, SIZE_TO_BYTES(flush_size - fpos));
  memset(temp_fingerprints + fpos, 0, flush_size - fpos);
  MEMCPY(node->data, temp_data,
         SIZE_TO_BYTES(flush_size),
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);
  MEMCPY(node->meta.fingerprints_,
         temp_fingerprints, flush_size,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

  NVM_BARRIER;

  uint64_t hdr = node->meta.header;
  SET_SIZE(hdr, flush_size);
  SET_ROWS(hdr, flush_rows);
  node->meta.header = hdr;
  PERSIST(node, 8);

  // update vlog bitmap
  vlog_manager_->UpdateBitmap(unused_indexes);
}

// For level 1, 4, 7..., we read key and modify prefixes in hash.
int32_t GlobalMemtable::ReadFromVLog(NVMNode* nvm_node, size_t level,
                                  uint64_t& leaf_vptr, uint64_t& leaf_hash,
                                  autovector<KVStruct>* split_buckets) {
  int32_t delta = 0;
  int32_t final_size = 0;

  auto data = nvm_node->data;
  for (size_t i = 0; i < NVM_MAX_SIZE; ++i) {
    KVStruct kv_info(data[(i << 1)], data[(i << 1) + 1]);
    if (!kv_info.vptr_) {
      continue;
    }

    Slice key;
    vlog_manager_->GetKey(kv_info.vptr_, key);

    if (key.size() == level) {
      final_size = kv_info.kv_size_;
      delta += final_size;
      leaf_vptr = kv_info.vptr_;
      leaf_hash = kv_info.hash_;
    } else {
      Rehash(key.data(), key.size(), kv_info.hash_, level + 1);
      split_buckets[static_cast<unsigned char>(key[level])]
          .push_back(kv_info);
    }
  }

  return final_size - delta;
}

int32_t GlobalMemtable::ReadFromNVM(NVMNode* nvm_node, size_t level,
                                 uint64_t& leaf_vptr, uint64_t& leaf_hash,
                                 autovector<KVStruct>* split_buckets) {
  int32_t delta = 0;
  int32_t final_size = 0;

  auto data = nvm_node->data;
  for (size_t i = 0; i < NVM_MAX_SIZE; ++i) {
    KVStruct kv_info(data[(i << 1)], data[(i << 1) + 1]);
    if (!kv_info.vptr_) {
      continue;
    }

    if (kv_info.key_length_ == level) {
      final_size = kv_info.kv_size_;
      delta += final_size;
      leaf_vptr = kv_info.vptr_;
      leaf_hash = kv_info.hash_;
    } else {
      split_buckets[static_cast<unsigned char>(GetPrefix(kv_info, level))]
          .push_back(kv_info);
    }
  }

  return final_size - delta;
}

void GlobalMemtable::SplitLeaf(InnerNode* leaf, size_t level,
                               InnerNode** node_need_split) {

  auto old_nvm_node = leaf->nvm_node_;
  *node_need_split = nullptr;

  uint64_t leaf_vptr = 0;
  uint64_t leaf_hash = 0;
  int32_t delta;
  autovector<KVStruct> split_buckets[256];
  if (level % 3 == 0) {
    delta = ReadFromVLog(old_nvm_node, level, leaf_vptr, leaf_hash, split_buckets);
  } else {
    delta = ReadFromNVM(old_nvm_node, level, leaf_vptr, leaf_hash, split_buckets);
  }
  leaf->heat_group_->UpdateSize(delta);

  int64_t oldest_key_time = leaf->oldest_key_time_;

  // leaf 0 is always created
  int32_t split_num = 1;
  for (size_t i = 1; i <= LAST_CHAR; ++i) {
    if (!split_buckets[i].empty()) {
      ++split_num;
    }
  }

  auto dummy_node = AllocateLeafNode(
      level + 1, static_cast<unsigned char>(LAST_CHAR), nullptr);
  dummy_node->parent_node_ = leaf;
  dummy_node->oldest_key_time_ = oldest_key_time;
  SET_NON_LEAF(dummy_node->status_);
  SET_TAG(dummy_node->nvm_node_->meta.header, DUMMY_TAG);
  PERSIST(dummy_node, 8);

  auto last_node = dummy_node;
  auto first_node = dummy_node;
  auto final_node = dummy_node;

  std::vector<InnerNode*> new_leaves;
  std::vector<unsigned char> prefixes;
  new_leaves.reserve(split_num + 1);
  prefixes.reserve(split_num);

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t temp_fingerprints[224] = {0};
  uint64_t temp_data[448] = {0};
#endif

  int h, bucket;
  uint8_t digit;

  for (int c = LAST_CHAR; c >= 0; --c) {
    if (split_buckets[c].empty() && c > 0) {
      continue;
    }
    auto new_leaf = AllocateLeafNode(
        level + 1, static_cast<unsigned char>(c), last_node);
    new_leaf->oldest_key_time_ = oldest_key_time;
    new_leaf->parent_node_ = leaf;
    auto nvm_node = new_leaf->nvm_node_;
    int pos = 0, fpos = 0;
    for (auto& kv_info : split_buckets[c]) {
      temp_fingerprints[fpos++] = static_cast<uint8_t>(kv_info.hash_);
      temp_data[pos++] = kv_info.hash_;
      temp_data[pos++] = kv_info.vptr_;
      new_leaf->estimated_size_ += kv_info.kv_size_;

      bucket = (int)(kv_info.hash_ & 63);
      h = (int)(kv_info.hash_ >> 6);
      digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
      new_leaf->hll_[bucket] = std::max(new_leaf->hll_[bucket], digit);
    }

    last_node = new_leaf;
    first_node = new_leaf;

    int flush_size = ALIGN_UP(fpos, 16);
    int flush_rows = SIZE_TO_ROWS(flush_size);
    assert(flush_rows <= NVM_MAX_ROWS);
    assert(flush_size <= NVM_MAX_SIZE);

    memset(temp_data + pos, 0, SIZE_TO_BYTES(flush_size - fpos));
    memset(temp_fingerprints + fpos, 0, flush_size - fpos);
    MEMCPY(nvm_node->data, temp_data, SIZE_TO_BYTES(flush_size),
           PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);
    MEMCPY(nvm_node->meta.fingerprints_, temp_fingerprints, flush_size,
           PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

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

  NVM_BARRIER;

  std::reverse(new_leaves.begin(), new_leaves.end());
  std::reverse(prefixes.begin(), prefixes.end());
  auto art = AllocateArtAfterSplit(new_leaves, prefixes);

  new_leaves.push_back(final_node);
  InsertSplitInnerNode(leaf, first_node, final_node, level + 1);
  leaf->estimated_size_ = 0;
  InsertNodesToGroup(leaf, new_leaves);

  {
    std::lock_guard<RWSpinLock> art_lk(leaf->art_rw_lock_);
    std::lock_guard<RWSpinLock> vptr_lk(leaf->vptr_lock_);

    leaf->art = art;
    leaf->vptr_ = leaf_vptr;
    leaf->hash_ = leaf_hash;
    SET_ART_NON_FULL(leaf->status_);
    SET_NON_LEAF(leaf->status_);
  }

}

// "4b prefix + 4b hash + 8b pointer" OR "4b SeqNum + 4b hash + 8b pointer"
// This function is responsible for unlocking opt_lock_
void GlobalMemtable::InsertIntoLeaf(InnerNode* leaf, KVStruct& kv_info,
                                    size_t level) {
  int write_pos = GET_NODE_BUFFER_SIZE(++leaf->status_) << 1;

  leaf->buffer_[write_pos - 2] = kv_info.hash_;
  leaf->buffer_[write_pos - 1] = kv_info.vptr_;
  leaf->estimated_size_ += kv_info.kv_size_;

  if (likely(write_pos < 32)) {
    leaf->opt_lock_.unlock();
    leaf->heat_group_->UpdateSize(kv_info.kv_size_);
    leaf->heat_group_->UpdateHeat();
    return;
  }

  leaf->heat_group_->UpdateSize(kv_info.kv_size_);
  leaf->heat_group_->UpdateHeat();

  InnerNode* next_to_split = nullptr;

  {
    // If we use read lock here, we can do concurrent read but block write,
    // if we use write lock here, we block read but allow concurrent write.
    shared_lock<SharedMutex> read_lk(leaf->share_mutex_);

    env_->GetCurrentTime(&leaf->oldest_key_time_);

    int rows = GET_ROWS(leaf->nvm_node_->meta.header);
    assert(rows < NVM_MAX_ROWS);

    FlushBuffer(leaf, rows);

    if (likely(rows < NVM_MAX_ROWS - 1)) {
      leaf->opt_lock_.unlock();
      return;
    }

    if (EstimateDistinctCount(leaf->hll_) < SQUEEZE_THRESHOLD) {
      SqueezeNode(leaf);
      leaf->opt_lock_.unlock();
      return;
    }

    SplitLeaf(leaf, level, &next_to_split);

    if (next_to_split) {
      next_to_split->opt_lock_.lock();
      next_to_split->share_mutex_.lock();
    }

    leaf->opt_lock_.unlock();
  }

  while (next_to_split) {
    InnerNode* current = next_to_split;
    SplitLeaf(current, ++level, &next_to_split);

    if (next_to_split) {
      next_to_split->opt_lock_.lock();
      next_to_split->share_mutex_.lock();
    }
    current->opt_lock_.unlock(false);
    current->share_mutex_.unlock();
  }
}

bool GlobalMemtable::ReadInNVMNode(NVMNode* nvm_node, uint64_t hash,
                                   std::string& key, std::string& value,
                                   Status* s) {
  ValueType type;
  uint64_t vptr;
  std::string found_key;

  for (size_t i = 0; i < 16; ++i) {
    if (!nvm_node->temp_buffer[i * 2 + 1]) {
      break;
    }

    if (nvm_node->temp_buffer[i * 2] != hash) {
      continue;
    }

    type = vlog_manager_->GetKeyValue(
        nvm_node->temp_buffer[i * 2 + 1], found_key, value);
    if (found_key == key) {
      *s = type == kTypeValue ? Status::OK() : Status::NotFound();
      return true;
    }
  }

  auto data = nvm_node->data;
  auto fingerprints = nvm_node->meta.fingerprints_;
  int rows = GET_ROWS(nvm_node->meta.header);

  int search_rows = (rows + 1) / 2 - 1;
  int size = rows * 16;
  __m256i target = _mm256_set1_epi8((uint8_t)hash);

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
      type = vlog_manager_->GetKeyValue(vptr, found_key, value);
      if (key == found_key) {
        *s = type == kTypeValue ? Status::OK() : Status::NotFound();
        return true;
      }
    }
  }

  return false;
}

bool GlobalMemtable::FindKeyInInnerNode(InnerNode* leaf, size_t level,
                                        std::string& key, std::string& value,
                                        Status* s) {
  shared_lock<SharedMutex> read_lk(leaf->share_mutex_);

  std::string found_key;
  ValueType type;

  uint64_t hash = HashAndPrefix(key.c_str(), key.length(), level);
  int pos = GET_NODE_BUFFER_SIZE(leaf->status_);
  auto buffer = leaf->buffer_;

  for (int i = 0; i < pos; ++i) {
    if (buffer[i * 2] != hash) {
      continue;
    }
    type = vlog_manager_->GetKeyValue(
        buffer[i * 2 + 1], found_key, value);
    if (found_key == key) {
      *s = type == kTypeValue ? Status::OK() : Status::NotFound();
      return true;
    }
  }

  auto nvm_node = leaf->nvm_node_;
  auto backup_nvm_node = leaf->backup_nvm_node_;
  bool need_search_backup = backup_nvm_node && backup_nvm_node != nvm_node;

  if (backup_nvm_node) {
    IncrementBackupRead();
  }

  bool found = ReadInNVMNode(nvm_node, hash, key, value, s);
  if (!found && need_search_backup) {
    found = ReadInNVMNode(backup_nvm_node, hash, key, value, s);
  }

  if (backup_nvm_node) {
    ReduceBackupRead();
  }

  return found;
}

InnerNode* GlobalMemtable::FindInnerNodeByKey(Slice& key,
                                              size_t& level,
                                              bool& stored_in_nvm) {
  size_t max_level = key.size();

  level = 1;
  stored_in_nvm = true;
  InnerNode* current = FindChild(root_, key[0]);

  while (current) {
    if (IS_LEAF(current->status_)) {
      return current;
    }

    if (level == max_level) {
      stored_in_nvm = false;
      return current;
    }

    current = FindChild(current, key[level++]);
  }

  return nullptr;
}

} // namespace ROCKSDB_NAMESPACE