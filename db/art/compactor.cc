//
// Created by joechen on 2022/4/3.
//

#include "compactor.h"

#include <iostream>

#include <db/db_impl/db_impl.h>

#include "utils.h"
#include "macros.h"
#include "nvm_node.h"
#include "heat_group.h"
#include "heat_group_manager.h"
#include "vlog_manager.h"
#include "global_memtable.h"
#include "node_allocator.h"

namespace ROCKSDB_NAMESPACE {

std::atomic<size_t> MemTotalSize;

void UpdateTotalSize(size_t update_size) {
  MemTotalSize.fetch_add(update_size, std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////

size_t Compactor::compaction_threshold_;

void Compactor::SetGroupManager(HeatGroupManager* group_manager) {
  group_manager_ = group_manager;
}

void Compactor::SetVLogManager(VLogManager* vlog_manager) {
  vlog_manager_ = vlog_manager;
}

void Compactor::SetDB(DBImpl* db_impl) {
  db_impl_ = db_impl;
}

void Compactor::Notify(HeatGroup* heat_group) {
  chosen_group_ = heat_group;
  cond_var_.notify_one();
}

void Compactor::StartCompactionThread() {
  compactor_thread_ =
      std::thread(&Compactor::BGWorkDoCompaction, this);
}

void Compactor::BGWorkDoCompaction() {
  while (true) {
    if (thread_stop_) {
      break;
    }

    if (MemTotalSize.load(std::memory_order_relaxed) < compaction_threshold_) {
      std::this_thread::sleep_for(std::chrono::milliseconds (100));
      continue;
    }

    std::unique_lock<std::mutex> lock{mutex_};
    group_manager_->AddOperation(nullptr, kOperationChooseCompaction, true);
    cond_var_.wait(lock);

    if (thread_stop_) {
      break;
    }

    if (chosen_group_) {
      DoCompaction();
      // Maybe cause bug here.
      chosen_group_->status_.store(kGroupWaitMove, std::memory_order_relaxed);
      group_manager_->AddOperation(chosen_group_, kOperatorMove, true);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds (100));
  }
}

void Compactor::StopCompactionThread() {
  thread_stop_ = true;
  cond_var_.notify_one();
  compactor_thread_.join();
}

void Compactor::DoCompaction() {
  //printf("Start compaction\n");

  InnerNode* start_node = chosen_group_->first_node_;
  InnerNode* end_node = chosen_group_->last_node_;
  InnerNode* start_node_backup = start_node;

  NVMNode* first_nvm_backup = GetNodeAllocator().AllocateNode();
  int32_t compacted_size = chosen_group_->group_size_.load(std::memory_order_relaxed);
  std::deque<InnerNode*> candidates;
  std::vector<NVMNode*> nvm_nodes;

  uint64_t num_entries = 0;
  uint64_t num_deletes = 0;
  uint64_t total_data_size = 0;
  int64_t oldest_key_time = start_node->oldest_key_time_;

  {
    std::lock_guard<std::mutex> lk(start_node->flush_mutex_);
    //std::lock_guard<std::mutex> gc_lk(start_node->gc_mutex_);

    if (IS_LEAF(start_node->status_)) {
      memcpy(first_nvm_backup, start_node->nvm_node_, PAGE_SIZE);
      start_node->oldest_key_time_ = LONG_LONG_MAX;
      start_node->estimated_size_ = 0;
      memset(start_node->hll_, 0, 64);
      nvm_nodes.push_back(first_nvm_backup);
    }
    start_node = start_node->next_node_;
  }

  ArtCompactionJob compaction_job;
  autovector<RecordIndex> compacted_records[VLogSegmentNum];

  while (start_node != end_node) {
    std::lock_guard<std::mutex> flush_lk(start_node->flush_mutex_);
    //std::lock_guard<std::mutex> gc_lk(start_node->gc_mutex_);

    if (unlikely(!IS_LEAF(start_node->status_))) {
      start_node = start_node->next_node_;
      continue;
    }

    auto new_nvm_node = GetNodeAllocator().AllocateNode();
    InsertNewNVMNode(start_node, new_nvm_node);
    candidates.push_back(start_node);
    nvm_nodes.push_back(start_node->backup_nvm_node_);

    oldest_key_time = std::min(oldest_key_time, start_node->oldest_key_time_);
    start_node->oldest_key_time_ = LONG_LONG_MAX;
    start_node->estimated_size_ = 0;
    memset(start_node->hll_, 0, 64);

    start_node = start_node->next_node_;
  }

  // Sort data
  SequenceNumber seq_num = 0;
  RecordIndex record_index = 0;

  std::vector<CompactionRec> kvs;
  kvs.reserve(NVM_MAX_SIZE);

  for (auto& nvm_node : nvm_nodes) {
    auto meta = nvm_node->meta;
    auto data = nvm_node->data;
    size_t size = GET_SIZE(meta.header);

    kvs.clear();
    for (size_t i = 0; i < size; ++i) {
      auto vptr = data[i * 2 + 1];
      if (!vptr) {
        continue;
      }

      std::string key, value;
      auto value_type = vlog_manager_->GetKeyValue(
          vptr, key, value, seq_num, record_index);
      kvs.emplace_back(key, value, seq_num, value_type);

      //compacted_records.push_back((kv_info.vptr_ & SegmentIDMask) | record_index);
      GetActualVptr(vptr);
      compacted_records[vptr >> 20].push_back(record_index);
    }

    std::stable_sort(kvs.begin(), kvs.end());
    std::string last_key;
    for (auto& kv : kvs) {
      if (kv.key != last_key) {
        last_key = kv.key;
        num_entries += kv.type == kTypeValue;
        num_deletes += kv.type == kTypeDeletion;
        total_data_size += (kv.key.length() + kv.value.length());
        compaction_job.compacted_data_.emplace_back(kv);
      }
    }
  }

  compaction_job.oldest_key_time_ = oldest_key_time;
  compaction_job.total_num_deletes_ = num_deletes;
  compaction_job.total_num_entries_ = num_entries;
  compaction_job.total_data_size_ = compaction_job.total_memory_usage_
      = total_data_size;

  // Compaction
  db_impl_->SyncCallFlush(&compaction_job);

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

  GetNodeAllocator().DeallocateNode(first_nvm_backup);

  for (size_t i = 0; i < VLogSegmentNum; ++i) {
    auto& vec = compacted_records[i];
    std::sort(vec.begin(), vec.end());
    auto header = (VLogSegmentHeader*)(vlog_manager_->pmemptr_ + VLogSegmentSize * i);
    if (!header->lock.mutex_.try_lock()) {
      continue;
    }
    auto bitmap = header->bitmap_;
    for (auto index : vec) {
      assert(index / 8 < VLogBitmapSize);
      bitmap[index / 8] &= ~(1 << (index % 8));
    }
    header->compacted_count_ += vec.size();
    // Just clwb, not use sfence
    // pmem_persist(header, VLogHeaderSize);

    header->lock.mutex_.unlock();
  }

  // Use sfence after all vlog are modified
  // _mm_sfence();

#ifndef NDEBUG
  printf("%p Compaction done: size_=%d, number of free pages: %zu\n",
         chosen_group_, compacted_size, GetNodeAllocator().GetNumFreePages());
#endif
  chosen_group_->UpdateSize(-compacted_size);
}

} // namespace ROCKSDB_NAMESPACE