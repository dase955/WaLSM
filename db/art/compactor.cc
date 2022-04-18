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

    vlog_manager_->FreeQueue();

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
  InnerNode* start_node = chosen_group_->first_node_;
  InnerNode* end_node = chosen_group_->last_node_;
  assert(IS_GROUP_START(start_node->status_));

  int32_t compacted_size = chosen_group_->group_size_.load(std::memory_order_relaxed);
  std::deque<InnerNode*> candidates;
  std::vector<NVMNode*> nvm_nodes;

  uint64_t num_entries = 0;
  uint64_t num_deletes = 0;
  uint64_t total_data_size = 0;
  int64_t oldest_key_time = start_node->oldest_key_time_;

  auto vlog_segment_num = vlog_manager_->vlog_segment_num_;
  auto vlog_segment_size = vlog_manager_->vlog_segment_size_;
  auto vlog_bitmap_size = vlog_manager_->vlog_bitmap_size_;

  auto node_after_end = end_node->next_node_;
  auto cur_node = start_node->next_node_;

  // Get candidate nodes
  while (cur_node != node_after_end) {
    std::lock_guard<std::mutex> flush_lk(cur_node->flush_mutex_);
    if (unlikely(!IS_LEAF(cur_node->status_) ||
                 cur_node->heat_group_ != chosen_group_)) {
      cur_node = cur_node->next_node_;
      continue;
    }

    auto new_nvm_node = GetNodeAllocator()->AllocateNode();
    InsertNewNVMNode(cur_node, new_nvm_node);
    candidates.push_back(cur_node);
    nvm_nodes.push_back(cur_node->backup_nvm_node_);

    oldest_key_time = std::min(oldest_key_time, cur_node->oldest_key_time_);
    cur_node->oldest_key_time_ = LONG_LONG_MAX;
    cur_node->estimated_size_ = 0;
    memset(cur_node->hll_, 0, 64);

    cur_node = cur_node->next_node_;
  }

  // Compaction
  ArtCompactionJob compaction_job;
  compaction_job.vlog_manager_ = vlog_manager_;
  compaction_job.nvm_nodes_ = std::vector<NVMNode*>(
      nvm_nodes.begin(), nvm_nodes.end());
  compaction_job.compacted_indexes_ =
      new std::vector<RecordIndex>[vlog_segment_num];
  db_impl_->SyncCallFlush(&compaction_job);

  // remove compacted node form list
  cur_node = start_node;
  auto candidate = candidates.front();
  candidates.pop_front();
  while (cur_node != node_after_end) {
    std::lock_guard<SpinMutex> lk(cur_node->link_lock_);
    if (cur_node->next_node_ == candidate) {
      RemoveOldNVMNode(cur_node);
      candidate = candidates.front();
      candidates.pop_front();
    }

    cur_node = cur_node->next_node_;
  }

  for (size_t i = 0; i < vlog_segment_num; ++i) {
    auto& indexes = compaction_job.compacted_indexes_[i];
    auto header = (VLogSegmentHeader*)(vlog_manager_->pmemptr_ + vlog_segment_size * i);
    if (!header->lock.mutex_.try_lock()) {
      continue;
    }
    if (header->status_ != kSegmentWritten) {
      header->lock.mutex_.unlock();
      continue;
    }
    auto bitmap = header->bitmap_;
    for (auto& index : indexes) {
      assert(index / 8 < vlog_bitmap_size);
      bitmap[index / 8] &= ~(1 << (index % 8));
    }
    header->compacted_count_ += indexes.size();
    PERSIST(header, vlog_manager_->vlog_header_size_);

    header->lock.mutex_.unlock();
  }

  // Use sfence after all vlog are modified
  MEMORY_BARRIER;

#ifndef NDEBUG
  printf("%p Compaction done: size_=%d, number of free pages: "
         "%zu, free vlog pages: %zu\n",
         chosen_group_, compacted_size,
         GetNodeAllocator()->GetNumFreePages(),
         vlog_manager_->free_pages_.size());
#endif
  chosen_group_->UpdateSize(-compacted_size);

  delete[] compaction_job.compacted_indexes_;
}

} // namespace ROCKSDB_NAMESPACE