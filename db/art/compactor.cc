//
// Created by joechen on 2022/4/3.
//

#include "compactor.h"

#include <condition_variable>
#include <cassert>
#include <thread>
#include <iostream>

#include "macros.h"
#include "nvm_node.h"
#include "heat_group.h"
#include "node_allocator.h"
#include "vlog_manager.h"
#include "global_memtable.h"

namespace ROCKSDB_NAMESPACE {

Compactor compactor;

Compactor &GetCompactor() {
  return compactor;
}

void Compactor::updateSize(size_t addedSize) {
  total_size_.fetch_add(addedSize, std::memory_order_relaxed);
}

void Compactor::MainLoop() {
  while (true) {
    if (m_bStop) {
      break;
    }

    if (total_size_.load(std::memory_order_relaxed) < CompactionThreshold) {
      std::this_thread::sleep_for(std::chrono::milliseconds (100));
      continue;
    }

    std::unique_lock<std::mutex> lock{mutex_};
    AddOperation(nullptr, kOperationChooseCompaction, false);
    cond_var_.wait(lock);

    if (m_bStop) {
      break;
    }

    if (chosen_group_) {
      total_size_.fetch_sub(doCompaction(), std::memory_order_release);
      AddOperation(chosen_group_, kOperatorMove, true);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds (100));
  }
}

void Compactor::StopLoop() {
  m_bStop = true;
  cond_var_.notify_one();
}

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

auto ReadAndSortNodeData(NVMNode *nvmNode) -> std::vector<CompactionRec> {
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
    GetManager().GetKeyValue(kvInfo.vptr, key, value, seq_num);
    kvs.emplace_back(key, value, seq_num);
  }

  // use stable sort to keep order of same key
  std::stable_sort(kvs.begin(), kvs.end());
  return kvs;
}

int32_t Compactor::doCompaction() {
  printf("Start compaction\n");

  InnerNode *startNode = chosen_group_->first_node_;
  InnerNode *endNode = chosen_group_->last_node_;
  InnerNode *startNodeBackup = startNode;

  NVMNode *firstNodeBackUp = nullptr;
  int32_t compactedSize;
  std::deque<InnerNode *> candidates;

  {
    std::lock_guard<std::mutex> lk(startNode->flush_mutex_);
    if (IS_LEAF(startNode->status_)) {
      firstNodeBackUp = GetNodeAllocator().AllocateNode();
      memcpy(firstNodeBackUp, startNode->nvm_node_, PAGE_SIZE);
      compactedSize = startNode->estimated_size_;
      startNode->estimated_size_ = 0;
    }
    startNode = startNode->next_node_;
  }


  while (startNode != endNode) {
    std::lock_guard<std::mutex> lk(startNode->flush_mutex_);
    if (!IS_LEAF(startNode->status_)) {
      startNode = startNode->next_node_;
      continue;
    }

    auto *newNVMNode = GetNodeAllocator().AllocateNode();
    InsertNewNVMNode(startNode, newNVMNode);
    candidates.push_back(startNode);

    compactedSize += startNode->estimated_size_;
    startNode->estimated_size_ = 0;

    startNode = startNode->next_node_;
  }

  // Compaction...

  // remove compacted node form list
  startNode = startNodeBackup;
  auto candidate = candidates.front();
  candidates.pop_front();
  while (startNode != endNode) {
    std::lock_guard<SpinLock> lk(startNode->link_lock_);
    if (startNode->next_node_ == candidate) {
      RemoveOldNVMNode(startNode);
      candidate = candidates.front();
      candidates.pop_front();
    }

    startNode = startNode->next_node_;
  }

  printf("Compaction done: size_=%d\n", compactedSize);
  chosen_group_->updateSize(-compactedSize);
  return compactedSize;
}

void Compactor::Notify(HeatGroup *heatGroup) {
  chosen_group_ = heatGroup;
  cond_var_.notify_one();
}

} // namespace ROCKSDB_NAMESPACE