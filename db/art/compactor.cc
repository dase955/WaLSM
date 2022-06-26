//
// Created by joechen on 2022/4/3.
//

#include "compactor.h"

#include <iostream>

#include <db/db_impl/db_impl.h>

#include "utils.h"
#include "logger.h"
#include "macros.h"
#include "nvm_node.h"
#include "heat_group.h"
#include "heat_group_manager.h"
#include "vlog_manager.h"
#include "global_memtable.h"
#include "node_allocator.h"
#include "art_node.h"

namespace ROCKSDB_NAMESPACE {

std::atomic<int64_t> MemTotalSize{0};
std::atomic<int64_t> SqueezedSize{0};
std::atomic<int64_t> CompactedSize{0};
std::atomic<int64_t> SqueezedSizeInCompaction{0};

std::atomic<int> BackupRead{0};

ThreadPool* SingleCompactionJob::thread_pool = NewThreadPool(4);

void UpdateTotalSize(int32_t update_size) {
  MemTotalSize.fetch_add(update_size, std::memory_order_release);
}

void UpdateTotalSqueezedSize(int64_t update_size) {
  SqueezedSize.fetch_add(update_size, std::memory_order_release);
}

void IncrementBackupRead() {
  BackupRead.fetch_add(1, std::memory_order_release);
}

void ReduceBackupRead() {
  BackupRead.fetch_sub(1, std::memory_order_release);
}

int64_t GetMemTotalSize() {
  return MemTotalSize.load(std::memory_order_acquire);
}

////////////////////////////////////////////////////////////

void RemoveChildren(InnerNode* parent, InnerNode* child) {
  // assert opt_lock and shared_mutex are held

  auto support_node = parent->support_node_;
  assert(GET_TAG(support_node->nvm_node_->meta.header, DUMMY_TAG));
  assert(!GET_TAG(support_node->nvm_node_->meta.header, GROUP_START_TAG));

  {
    auto heat_group = parent->heat_group_;
    std::lock_guard<RWSpinLock> link_lk(parent->link_lock_);
    std::lock_guard<std::mutex> heat_lk(heat_group->lock);

    InnerNode* next_node = support_node->next_node_;

    if (support_node == heat_group->last_node_) {
      heat_group->last_node_ = parent;
    } else if (child == heat_group->last_node_) {
      heat_group->last_node_ = parent;
      next_node = child->next_node_;

      assert(IS_GROUP_START(next_node));
      assert(next_node->next_node_ == support_node);

      auto after = support_node->next_node_;
      std::lock_guard<RWSpinLock> next_link_lk(after->link_lock_);

      next_node->next_node_ = after;
      auto next_nvm_node = GetNodeAllocator()->relative(
          after->backup_nvm_node_
              ? after->backup_nvm_node_ : after->nvm_node_);

      auto& nvm_meta = next_node->nvm_node_->meta;
      if (GET_TAG(nvm_meta.header, ALT_FIRST_TAG)) {
        nvm_meta.next1 = next_nvm_node;
      } else {
        nvm_meta.next2 = next_nvm_node;
      }
      PERSIST(next_node->nvm_node_, CACHE_LINE_SIZE);
    }

    {
      std::lock_guard<RWSpinLock> next_link_lk(next_node->link_lock_);
      parent->next_node_ = next_node;
      auto next_nvm_node =
          GetNodeAllocator()->relative(next_node->backup_nvm_node_
                                           ? next_node->backup_nvm_node_ : next_node->nvm_node_);
      auto old_hdr = parent->nvm_node_->meta.header;
      auto new_hdr = old_hdr;
      if (GET_TAG(old_hdr, ALT_FIRST_TAG)) {
        CLEAR_TAG(new_hdr, ALT_FIRST_TAG);
        parent->nvm_node_->meta.next2 = next_nvm_node;
      } else {
        SET_TAG(new_hdr, ALT_FIRST_TAG);
        parent->nvm_node_->meta.next1 = next_nvm_node;
      }
      SET_ROWS(new_hdr, 0);
      SET_SIZE(new_hdr, 0);
      parent->nvm_node_->meta.header = new_hdr;
      parent->support_node_ = parent;
      PERSIST(parent->nvm_node_, 8);
    }

    GetNodeAllocator()->DeallocateNode(support_node->nvm_node_);
    delete support_node;
  }

  {
    std::lock_guard<RWSpinLock> art_lk(parent->art_rw_lock_);
    std::lock_guard<RWSpinLock> vptr_lk(parent->vptr_lock_);

    parent->backup_art = parent->art;
    parent->art = nullptr;

    SET_LEAF(parent);
    SET_ART_NON_FULL(parent);
    SET_NODE_BUFFER_SIZE(parent->status_, 0);
    memset(parent->buffer_, 0, 256);
    parent->estimated_size_ = 0;
    if (parent->vptr_) {
      ++parent->status_;
      parent->estimated_size_ = parent->vptr_ >> 48;
      parent->buffer_[0] = parent->hash_;
      parent->buffer_[1] = parent->vptr_;
    }
    parent->hash_ = parent->vptr_ = 0;
  }
}

void ProcessNodes(InnerNode* parent, std::vector<InnerNode*>& children,
                  SingleCompactionJob* job) {
  std::lock_guard<OptLock> opt_lk(parent->opt_lock_);
  std::lock_guard<SharedMutex> write_lk(parent->share_mutex_);

  if (children.size() == parent->art->num_children_ && parent->heat_group_ == job->group_) {
    RemoveChildren(parent, children.back());
    job->candidate_parents.push_back(parent);
    for (auto child : children) {
      assert(child->heat_group_ == job->group_);
      assert(child->heat_group_ == parent->heat_group_);
      SET_NODE_INVALID(child->status_);
      job->candidates_removed.push_back(child);
      child->opt_lock_.unlock();
    }
  } else {
    for (auto child : children) {
      job->candidates.push_back(child);
      child->opt_lock_.unlock();
    }
  }
}

////////////////////////////////////////////////////////////

int64_t Compactor::compaction_threshold_;

size_t Compactor::max_rewrite_count;

int Compactor::rewrite_threshold;

Compactor::Compactor(const DBOptions& options)
    : group_manager_(nullptr), vlog_manager_(nullptr),
      num_parallel_compaction_(options.num_parallel_compactions) {}

Compactor::~Compactor() noexcept {
  printf("Statistics:\n"
      "Data remain in nvm:   %zu\n"
      "Total compacted size: %zu\n"
      "Total squeezed size:  %zu\n",
      MemTotalSize.load(std::memory_order_relaxed),
      CompactedSize.load(std::memory_order_relaxed),
      SqueezedSize.load(std::memory_order_relaxed));

  for (auto job : compaction_jobs_) {
    delete[] job->compacted_indexes;
    delete job;
  }
}

void Compactor::SetGroupManager(HeatGroupManager* group_manager) {
  group_manager_ = group_manager;
}

void Compactor::SetGlobalMemtable(GlobalMemtable* global_memtable) {
  global_memtable_ = global_memtable;
}

void Compactor::SetVLogManager(VLogManager* vlog_manager) {
  vlog_manager_ = vlog_manager;
  for (int i = 0; i < num_parallel_compaction_; ++i) {
    auto job = new SingleCompactionJob;
    job->keys_in_node = std::vector<std::string>(240);
    job->compacted_indexes =
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

void Compactor::RewriteData(SingleCompactionJob* job) {
  job->keys_in_node[0].clear();

  for (auto vptr : job->hot_data) {
    int i = vptr >> 20;
    auto header = (VLogSegmentHeader*)(
        vlog_manager_->pmemptr_ + vlog_manager_->vlog_segment_size_ * i);

    char* record_start = vlog_manager_->pmemptr_ + vptr;
    Slice slice(record_start, 1 << 20);
    ValueType type = *((ValueType*)record_start);
    slice.remove_prefix(WriteBatchInternal::kRecordPrefixSize);

    uint32_t key_length;
    GetVarint32(&slice, &key_length);
    Slice key(slice.data(), key_length);
    slice.remove_prefix(key_length);

    if (type == kTypeValue) {
      uint32_t value_length;
      GetVarint32(&slice, &value_length);
      slice.remove_prefix(value_length);
    }

    auto kv_size = slice.data() - record_start;
    std::lock_guard<SpinMutex> status_lk(header->lock.mutex_);
    if (unlikely(header->status_ == kSegmentGC)) {
      std::string record(record_start, kv_size);
      vlog_manager_->WriteToNewSegment(record, vptr);
    }

    auto hash = HashOnly(key.data(), key.size());
    KVStruct kv_info(hash, vptr);
    kv_info.kv_size_ = kv_size;
    global_memtable_->Put(key, kv_info, false);
  }
}

void Compactor::BGWork() {
  static int64_t choose_threshold = compaction_threshold_ +
                                    num_parallel_compaction_ *
                                        HeatGroup::group_min_size_;

  while (true) {
    if (thread_stop_) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    vlog_manager_->FreeQueue();
    GetNodeAllocator()->FreeNodes();
    for (auto job : chosen_jobs_) {
      for (auto art : job->removed_arts) {
        DeleteArtNode(art);
      }
    }
    chosen_jobs_.clear();

    auto cur_mem_size = MemTotalSize.load(std::memory_order_relaxed);
    if (cur_mem_size < choose_threshold) {
      continue;
    }

    {
      std::unique_lock<std::mutex> lock{mutex_};
      group_manager_->AddOperation(nullptr, kOperationChooseCompaction, false, this);
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

    auto start_time = GetStartTime();

    SingleCompactionJob::thread_pool->SetJobCount(chosen_jobs_.size());
    for (auto& job : chosen_jobs_) {
      auto func = std::bind(&Compactor::CompactionPreprocess, this, job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    db_impl_->SyncCallFlush(chosen_jobs_);

    uint64_t total_out_size = 0;
    SingleCompactionJob::thread_pool->SetJobCount(chosen_jobs_.size());
    for (auto job : chosen_jobs_) {
      total_out_size += job->out_file_size;
      auto func = std::bind(&Compactor::CompactionPostprocess, this, job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    for (auto job : chosen_jobs_) {
      group_manager_->AddOperation(job->group_, kOperatorMove, true);
    }

    auto end_time = GetStartTime();

    // Use sfence after all vlog are modified
    NVM_BARRIER;

    RECORD_INFO("%ld, %.2fMB, %.2fMB, %.3lf, %.3lf, %.5fs, %.3fs, %ld\n",
                0, total_out_size / 1048576.0,
                total_out_size / 1048576.0, 1.0, 1.0,
                (end_time - start_time) * 1e-6, start_time * 1e-6,
                0);

    int rewrite_count = 0;
    int total_count = 0;
    int hot_count = 0;
    int recent_count = 0;
    for (auto job : chosen_jobs_) {
      recent_count += job->recent_count;
      rewrite_count += job->hot_data.size();
      total_count += job->total_count;
      hot_count += job->hot_count;
    }

    auto squeezed_in_compaction =
        (float)SqueezedSizeInCompaction.load(std::memory_order_relaxed) / 1048576.0f;
    auto mem_total_size =
        (float)MemTotalSize.load(std::memory_order_relaxed) / 1048576.0f;
    auto compacted_size =
        (float)CompactedSize.load(std::memory_order_relaxed) / 1048576.0f;
    auto total_squeezed_size =
        (float)SqueezedSize.load(std::memory_order_relaxed) / 1048576.0f;

    RECORD_DEBUG("free pages: "
        "%zu, %zu(%.2f), size: %.2fM, %.2fM, %.2fM; squeezed in groups %.2fM "
        "rewrite %d total count %d hot count %d, recent count %d\n",
        GetNodeAllocator()->GetNumFreePages(),
        vlog_manager_->free_segments_.size(), vlog_manager_->Estimate(),
        mem_total_size, compacted_size, total_squeezed_size,
        squeezed_in_compaction,
        rewrite_count, total_count, hot_count, recent_count);

    SqueezedSizeInCompaction.store(0);

    // Wait all reading in backup nvm node finish,
    // then we can free vlog pages and nodes.
    while (BackupRead.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    SingleCompactionJob::thread_pool->SetJobCount(chosen_jobs_.size());
    for (auto& job : chosen_jobs_) {
      auto func = std::bind(&Compactor::RewriteData, this, job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();
  }
}

void Compactor::CompactionPreprocess(SingleCompactionJob* job) {
  auto chosen_group = job->group_;
  InnerNode* start_node = chosen_group->first_node_;
  InnerNode* next_start_node = chosen_group->next_seq->first_node_;
  assert(IS_GROUP_START(start_node));
  assert(IS_GROUP_START(next_start_node));

  job->Reset();
  job->vlog_manager_ = vlog_manager_;

  int32_t compacted_size = 0;
  int64_t squeezed_size = 0;
  std::vector<InnerNode*> children;
  InnerNode* cur_parent = nullptr;

  InnerNode* cur_node = start_node->next_node_;
  while (cur_node != next_start_node) {
    cur_node->opt_lock_.lock();
    cur_node->share_mutex_.lock();

    if (NOT_LEAF(cur_node) || !cur_node->heat_group_) {
      cur_node->share_mutex_.unlock();
      cur_node->opt_lock_.unlock();
      cur_node = cur_node->next_node_;
      continue;
    }

    job->nvm_nodes_and_sizes.emplace_back(
        cur_node->nvm_node_, GET_SIZE(cur_node->nvm_node_->meta.header));
    job->oldest_key_time_ =
        std::min(job->oldest_key_time_, cur_node->oldest_key_time_);

    MEMCPY(cur_node->nvm_node_->temp_buffer, cur_node->buffer_,
           SIZE_TO_BYTES(GET_NODE_BUFFER_SIZE(cur_node->status_)),
           PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);
    SET_NODE_BUFFER_SIZE(cur_node->status_, 0);
    auto new_nvm_node = GetNodeAllocator()->AllocateNode();
    InsertNewNVMNode(cur_node, new_nvm_node);
    compacted_size += cur_node->estimated_size_;
    squeezed_size += cur_node->squeezed_size_;

    cur_node->estimated_size_ = 0;
    cur_node->squeezed_size_ = 0;
    memset(cur_node->hll_, 0, 64);

    cur_node->share_mutex_.unlock();

    if (cur_parent && cur_node->parent_node_ != cur_parent) {
      ProcessNodes(cur_parent, children, job);
      children.clear();
    }

    children.push_back(cur_node);
    cur_parent = cur_node->parent_node_;
    cur_node = cur_node->next_node_;
  }

  NVM_BARRIER;

  if (cur_parent && !children.empty()) {
    ProcessNodes(cur_parent, children, job);
  }

  job->candidates.push_back(nullptr);
  job->group_->group_size_.fetch_add(-compacted_size, std::memory_order_relaxed);
  UpdateTotalSize(-compacted_size);
  SqueezedSizeInCompaction.fetch_add(squeezed_size, std::memory_order_release);
  CompactedSize.fetch_add(compacted_size, std::memory_order_release);
}

void Compactor::CompactionPostprocess(SingleCompactionJob* job) {
  // remove compacted node form list
  auto cur_node = job->group_->first_node_;
  auto node_after_end = job->group_->last_node_->next_node_;
  auto candidates = job->candidates;
  assert(IS_GROUP_START(cur_node));

  auto candidate = candidates.front();
  candidates.pop_front();
  while (candidate && cur_node != node_after_end) {
    std::lock_guard<RWSpinLock> link_lk(cur_node->link_lock_);
    if (cur_node->next_node_ == candidate) {
      RemoveOldNVMNode(cur_node);
      candidate = candidates.front();
      candidates.pop_front();
    }
    cur_node = cur_node->next_node_;
  }

  assert(candidates.empty());

  for (auto removed : job->candidates_removed) {
    GetNodeAllocator()->DeallocateNode(removed->backup_nvm_node_);
    removed->backup_nvm_node_ = nullptr;
  }

  for (auto parent : job->candidate_parents) {
    assert(parent->backup_art);
    std::lock_guard<RWSpinLock> art_lk(parent->art_rw_lock_);
    job->removed_arts.push_back(parent->backup_art);
    parent->backup_art = nullptr;
  }

  vlog_manager_->UpdateBitmap(job->compacted_indexes);

  job->group_->status_.store(kGroupWaitMove, std::memory_order_relaxed);
}

} // namespace ROCKSDB_NAMESPACE