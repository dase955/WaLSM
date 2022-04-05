//
// Created by joechen on 2022/4/3.
//

#include "compactor.h"

#include <iostream>

#include "utils.h"
#include "macros.h"
#include "nvm_node.h"
#include "heat_group.h"
#include "heat_group_manager.h"
#include "node_allocator.h"
#include "vlog_manager.h"
#include "global_memtable.h"

namespace ROCKSDB_NAMESPACE {

struct CompactionRec {
  std::string key;
  std::string value;
  uint64_t    seqNum;

  CompactionRec() = default;

  CompactionRec(std::string &key_, std::string &value_, uint64_t seqNum_ )
      : key(key_), value(value_), seqNum(seqNum_){};

  friend bool operator<(const CompactionRec& l, const CompactionRec& r) {
    return l.key < r.key;
  }
};

std::atomic<size_t> MemTotalSize;

void UpdateTotalSize(size_t update_size) {
  MemTotalSize.fetch_add(update_size, std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////

void Compactor::SetGroupManager(HeatGroupManager *group_manager) {
  group_manager_ = group_manager;
}

void Compactor::SetVLogManager(VLogManager *vlog_manager) {
  vlog_manager_ = vlog_manager;
}

void Compactor::Notify(HeatGroup *heat_group) {
  chosen_group_ = heat_group;
  cond_var_.notify_one();
}

void Compactor::BGWorkDoCompaction() {
  while (true) {
    if (thread_stop_) {
      break;
    }

    if (MemTotalSize.load(std::memory_order_relaxed) < CompactionThreshold) {
      std::this_thread::sleep_for(std::chrono::milliseconds (100));
      continue;
    }

    std::unique_lock<std::mutex> lock{mutex_};
    group_manager_->AddOperation(nullptr, kOperationChooseCompaction, false);
    cond_var_.wait(lock);

    if (thread_stop_) {
      break;
    }

    if (chosen_group_) {
      MemTotalSize.fetch_sub(DoCompaction(), std::memory_order_release);
      group_manager_->AddOperation(chosen_group_, kOperatorMove, true);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds (100));
  }
}

void Compactor::StopBGWork() {
  thread_stop_ = true;
  cond_var_.notify_one();
}

int32_t Compactor::DoCompaction() {
  printf("Start compaction\n");

  InnerNode *start_node = chosen_group_->first_node_;
  InnerNode *end_node = chosen_group_->last_node_;
  InnerNode *start_node_backup = start_node;

  NVMNode *first_nvm_backup = nullptr;
  int32_t compacted_size = 0;
  std::deque<InnerNode *> candidates;

  {
    std::lock_guard<std::mutex> lk(start_node->flush_mutex_);
    if (IS_LEAF(start_node->status_)) {
      first_nvm_backup = GetNodeAllocator().AllocateNode();
      memcpy(first_nvm_backup, start_node->nvm_node_, PAGE_SIZE);
      compacted_size = start_node->estimated_size_;
      start_node->estimated_size_ = 0;
    }
    start_node = start_node->next_node_;
  }


  while (start_node != end_node) {
    std::lock_guard<std::mutex> lk(start_node->flush_mutex_);
    if (!IS_LEAF(start_node->status_)) {
      start_node = start_node->next_node_;
      continue;
    }

    auto *new_nvm_node = GetNodeAllocator().AllocateNode();
    InsertNewNVMNode(start_node, new_nvm_node);
    candidates.push_back(start_node);

    compacted_size += start_node->estimated_size_;
    start_node->estimated_size_ = 0;

    start_node = start_node->next_node_;
  }

  // Compaction...

  // remove compacted node form list
  start_node = start_node_backup;
  auto candidate = candidates.front();
  candidates.pop_front();
  while (start_node != end_node) {
    std::lock_guard<SpinMutex> lk(start_node->link_lock_);
    if (start_node->next_node_ == candidate) {
      RemoveOldNVMNode(start_node);
      candidate = candidates.front();
      candidates.pop_front();
    }

    start_node = start_node->next_node_;
  }

  printf("Compaction done: size_=%d\n", compacted_size);
  chosen_group_->UpdateSize(-compacted_size);
  return compacted_size;
}

void Compactor::ReadAndSortNodeData(NVMNode *nvmNode) {
  auto meta = nvmNode->meta;
  auto data = nvmNode->data;
  size_t size = GET_SIZE(meta.header);
  std::vector<CompactionRec> kvs;
  kvs.reserve(NVM_MAX_SIZE);
  for (size_t i = 0; i < size; ++i) {
    KVStruct kvInfo{data[i * 2], data[i * 2 + 1]};
    if (kvInfo.vptr == 0) {
      continue;
    }

    uint64_t seq_num;
    std::string key, value;
    vlog_manager_->GetKeyValue(kvInfo.vptr, key, value, seq_num);
    kvs.emplace_back(key, value, seq_num);
  }

  // use stable sort to keep order of same key
  std::stable_sort(kvs.begin(), kvs.end());
}

} // namespace ROCKSDB_NAMESPACE