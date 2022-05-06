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

std::atomic<int64_t> MemTotalSize{0};

ThreadPool* SingleCompactionJob::thread_pool = NewThreadPool(4);

void UpdateTotalSize(int32_t update_size) {
  MemTotalSize.fetch_add(update_size, std::memory_order_relaxed);
}

int64_t GetMemTotalSize() {
  return MemTotalSize.load(std::memory_order_relaxed);
}

////////////////////////////////////////////////////////////

int64_t Compactor::compaction_threshold_;

Compactor::~Compactor() noexcept {
  for (auto job : compaction_jobs_) {
    delete[] job->compacted_indexes_;
    delete job;
  }
}

void Compactor::SetGroupManager(HeatGroupManager* group_manager) {
  group_manager_ = group_manager;
}

void Compactor::SetVLogManager(VLogManager* vlog_manager) {
  vlog_manager_ = vlog_manager;
  for (int i = 0; i < num_parallel_compaction_; ++i) {
    auto job = new SingleCompactionJob;
    job->compacted_indexes_ =
        new autovector<RecordIndex>[vlog_manager_->vlog_segment_num_];
    compaction_jobs_.push_back(job);
  }
}

void Compactor::SetDB(DBImpl* db_impl) {
  db_impl_ = db_impl;
}

void Compactor::Notify(std::vector<HeatGroup*>& heat_groups) {
  std::unique_lock<std::mutex> lock{mutex_};
  chosen_groups_ = std::move(heat_groups);
  cond_var_.notify_one();
}

void Compactor::StartCompactionThread() {
  compactor_thread_ =
      std::thread(&Compactor::BGWorkDoCompaction, this);
}

void Compactor::StopCompactionThread() {
  thread_stop_ = true;
  cond_var_.notify_one();
  compactor_thread_.join();
}

void Compactor::CompactionPreprocess(SingleCompactionJob* job) {
  auto chosen_group = job->group_;
  InnerNode* start_node = chosen_group->first_node_;
  InnerNode* end_node = chosen_group->last_node_;
  InnerNode* node_after_end = end_node->next_node_;
  InnerNode* cur_node = start_node->next_node_;
  assert(IS_GROUP_START(start_node->status_));

  job->candidates_.clear();
  job->nvm_nodes_.clear();
  job->start_node_ = start_node;
  job->node_after_end_ = node_after_end;
  job->vlog_manager_ = vlog_manager_;

  // Get candidate nodes
  int compacted_size = 0;
  while (cur_node != node_after_end) {
    auto& opt_lock = cur_node->opt_lock_;
    opt_lock.WriteLock();

    std::lock_guard<std::mutex> flush_lk(cur_node->flush_mutex_);
    if (!(IS_LEAF(cur_node->status_) &&
          cur_node->heat_group_ == chosen_group) ||
        (GET_ROWS(cur_node->nvm_node_->meta.header) == 0 &&
         GET_NODE_BUFFER_SIZE(cur_node->status_) == 0)) {
      cur_node = cur_node->next_node_;
      opt_lock.WriteUnlock(true);
      continue;
    }

    // Also flush data in buffer
    MEMCPY(cur_node->nvm_node_->temp_buffer, cur_node->buffer_,
           SIZE_TO_BYTES(GET_NODE_BUFFER_SIZE(cur_node->status_)),
           PMEM_F_MEM_NODRAIN);
    SET_NODE_BUFFER_SIZE(cur_node->status_, 0);
    compacted_size += cur_node->estimated_size_;
    cur_node->estimated_size_ = 0;
    opt_lock.WriteUnlock(false);

    auto new_nvm_node = GetNodeAllocator()->AllocateNode();
    InsertNewNVMNode(cur_node, new_nvm_node);
    job->candidates_.push_back(cur_node);
    job->nvm_nodes_.push_back(cur_node->backup_nvm_node_);

    job->oldest_key_time_ =
        std::min(job->oldest_key_time_, cur_node->oldest_key_time_);
    cur_node->oldest_key_time_ = LONG_LONG_MAX;
    memset(cur_node->hll_, 0, 64);

    cur_node = cur_node->next_node_;
  }

  job->candidates_.push_back(nullptr);
  job->group_->group_size_.fetch_add(-compacted_size, std::memory_order_relaxed);
  UpdateTotalSize(-compacted_size);
}

void Compactor::CompactionPostprocess(SingleCompactionJob* job) {
  // remove compacted node form list
  auto cur_node = job->start_node_;
  auto node_after_end = job->node_after_end_;
  auto candidates = job->candidates_;

  auto vlog_segment_num = vlog_manager_->vlog_segment_num_;
  auto vlog_segment_size = vlog_manager_->vlog_segment_size_;
  [[maybe_unused]] auto vlog_bitmap_size = vlog_manager_->vlog_bitmap_size_;

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

  assert(candidates.empty());

  for (size_t i = 0; i < vlog_segment_num; ++i) {
    auto& indexes = job->compacted_indexes_[i];
    auto header = (VLogSegmentHeader*)(
        vlog_manager_->pmemptr_ + vlog_segment_size * i);

    if (!header->lock.mutex_.try_lock()) {
      indexes.clear();
      continue;
    }

    if (header->status_ != kSegmentWritten) {
      header->lock.mutex_.unlock();
      indexes.clear();
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

    indexes.clear();
  }

  job->group_->status_.store(kGroupWaitMove, std::memory_order_relaxed);
  group_manager_->AddOperation(job->group_, kOperatorMove, true);
}

void Compactor::BGWorkDoCompaction() {
  while (true) {
    if (thread_stop_) {
      break;
    }

    vlog_manager_->FreeQueue();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto mem_total_size = MemTotalSize.load(std::memory_order_relaxed);
    if (mem_total_size < compaction_threshold_) {
      continue;
    }

    {
      std::unique_lock<std::mutex> lock{mutex_};
      group_manager_->AddOperation(nullptr, kOperationChooseCompaction, true);
      cond_var_.wait(lock);

      if (thread_stop_) {
        return;
      }

      if (chosen_groups_.empty()) {
        continue;
      }

      for (size_t i = 0; i < chosen_groups_.size(); ++i) {
        auto job = compaction_jobs_[i];
        job->group_ = chosen_groups_[i];
        chosen_jobs_.push_back(job);
      }
    }

    SingleCompactionJob::thread_pool->SetJobCount(chosen_jobs_.size());
    for (auto& job : chosen_jobs_) {
      auto func = std::bind(&Compactor::CompactionPreprocess, this, job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    db_impl_->SyncCallFlush(chosen_jobs_);

    SingleCompactionJob::thread_pool->SetJobCount(chosen_jobs_.size());
    for (auto& job : chosen_jobs_) {
      auto func = std::bind(&Compactor::CompactionPostprocess, this, job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    // Use sfence after all vlog are modified
    MEMORY_BARRIER;

//#ifndef NDEBUG
   float estimate = vlog_manager_->Estimate();
    printf("%d Compaction done, number of free pages: "
        "%zu, free vlog pages: %zu, free ratio:%f, size:%zu\n",
        (int)chosen_jobs_.size(),
        GetNodeAllocator()->GetNumFreePages(),
        vlog_manager_->free_segments_.size(), estimate,
        MemTotalSize.load(std::memory_order_relaxed));
//#endif

    chosen_jobs_.clear();
  }
}

} // namespace ROCKSDB_NAMESPACE