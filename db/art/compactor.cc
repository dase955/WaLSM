//
// Created by joechen on 2022/4/3.
//

#include "compactor.h"

#include <iostream>
#include <unistd.h>

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
std::atomic<int>     BackupRead{0};
PriorityLock         IteratorLock;


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

void TryCreateIterator() {
  IteratorLock.LockLowPri();
}

void DeleteIterator() {
  IteratorLock.UnlockLowPri();
}

#ifdef ROCKSDB_SUPPORT_THREAD_LOCAL
thread_local uint8_t fingerprints[224] = {0};
thread_local uint64_t nvm_data[448] = {0};
#endif

struct alignas(CACHE_LINE_SIZE) CompactionRec {
  std::string* key;
  Slice        value_slice;
  uint64_t     seq_num;
  RecordIndex  record_index;
  KVStruct     s;

  CompactionRec() = default;

  friend bool operator<(const CompactionRec& l, const CompactionRec& r) {
    return *l.key < *r.key;
  }

  friend bool operator==(const CompactionRec& l, const CompactionRec& r) {
    return *l.key == *r.key;
  }
};

////////////////////////////////////////////////////////////

void RewriteData(std::vector<KVStruct>& rewrite_kv, InnerNode* node,
                 VLogManager* vlog_manager) {
  if (rewrite_kv.empty()) {
    return;
  }

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t fingerprints[224] = {0};
  uint64_t nvm_data[448] = {0};
#endif

  int pos = 0, fpos = 0;
  int h, bucket;
  uint8_t digit;
  auto nvm_node = node->nvm_node_;

  for (auto& kv_info : rewrite_kv) {
    vlog_manager->MaybeRewrite(kv_info);
    fingerprints[fpos++] = static_cast<uint8_t>(kv_info.actual_hash);
    nvm_data[pos++] = kv_info.hash;
    nvm_data[pos++] = kv_info.vptr;
    node->estimated_size_ += kv_info.kv_size;

    bucket = (int)(kv_info.actual_hash & 63);
    h = (int)(kv_info.actual_hash >> 6);
    digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
    node->hll_[bucket] = std::max(node->hll_[bucket], digit);
  }

  int flush_size = ALIGN_UP(fpos, 16);
  int flush_rows = SIZE_TO_ROWS(flush_size);
  assert(flush_rows <= NVM_MAX_ROWS);
  assert(flush_size <= NVM_MAX_SIZE);

  memset(nvm_data + pos, 0, SIZE_TO_BYTES(flush_size - fpos));
  memset(fingerprints + fpos, 0, flush_size - fpos);
  MEMCPY(nvm_node->data, nvm_data, SIZE_TO_BYTES(flush_size),
         PMEM_F_MEM_NONTEMPORAL);
  NVM_BARRIER;
  MEMCPY(nvm_node->meta.fingerprints_, fingerprints, flush_size,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

  auto hdr = nvm_node->meta.header;
  SET_SIZE(hdr, flush_size);
  SET_ROWS(hdr, flush_rows);
  nvm_node->meta.header = hdr;
  PERSIST(nvm_node, CACHE_LINE_SIZE);
}

void RemoveChildren(InnerNode* parent, InnerNode* child) {
  // assert opt_lock and shared_mutex are held

  auto support_node = parent->support_node;
  assert(GET_TAG(support_node->nvm_node_->meta.header, DUMMY_TAG));
  assert(!GET_TAG(support_node->nvm_node_->meta.header, GROUP_START_TAG));

  {
    auto heat_group = parent->heat_group_;
    std::lock_guard<RWSpinLock> link_lk(parent->link_lock_);
    std::lock_guard<std::mutex> heat_lk(heat_group->lock);

    InnerNode* next_node = support_node->next_node;

    if (support_node == heat_group->last_node_) {
      heat_group->last_node_ = parent;
    } else if (child == heat_group->last_node_) {
      heat_group->last_node_ = parent;
      next_node = child->next_node;

      assert(IS_GROUP_START(next_node));
      assert(next_node->next_node == support_node);

      auto after = support_node->next_node;
      std::lock_guard<RWSpinLock> next_link_lk(after->link_lock_);

      next_node->next_node = after;
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
      parent->next_node = next_node;
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
      parent->support_node = parent;
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
                  SingleCompactionJob* job, std::vector<KVStruct>& rewrite_kv) {
  std::lock_guard<OptLock> opt_lk(parent->opt_lock_);
  std::lock_guard<SharedMutex> write_lk(parent->share_mutex_);

  if (children.size() == parent->art->num_children_ &&
      parent->heat_group_ == job->group_ &&
      rewrite_kv.size() < BULK_WRITE_SIZE) {
    RemoveChildren(parent, children.back());
    RewriteData(rewrite_kv, parent, job->vlog_manager_);
    job->candidate_parents.push_back(parent);
    for (auto child : children) {
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

Compactor::Compactor(const DBOptions& options)
    : group_manager_(nullptr), vlog_manager_(nullptr),
      num_parallel_compaction_(options.num_parallel_compactions),
      rewrite_threshold_(options.enable_rewrite ? 1 : INT_MAX) {}

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

void Compactor::SetVLogManager(VLogManager* vlog_manager) {
  vlog_manager_ = vlog_manager;

  all_compacted_indexes_.resize(vlog_manager_->vlog_segment_num_);
  for (auto& indexes : all_compacted_indexes_) {
    indexes.reserve(16);
  }

  for (int i = 0; i < num_parallel_compaction_; ++i) {
    auto job = new SingleCompactionJob;
    job->keys_in_node = std::vector<std::string>(241);
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

std::atomic<int> total_rewrite{0};

void Compactor::BGWork() {
  static int64_t choose_threshold = compaction_threshold_ +
                                    num_parallel_compaction_ *
                                        HeatGroup::group_min_size_;

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    vlog_manager_->FreeQueue();
    GetNodeAllocator()->FreeNodes();
    for (auto job : chosen_jobs_) {
      for (auto art : job->removed_arts) {
        DeleteArtNode(art);
      }
    }
    chosen_jobs_.clear();

    if (thread_stop_) {
      break;
    }

    auto cur_mem_size = MemTotalSize.load(std::memory_order_relaxed);
    if (cur_mem_size < choose_threshold) {
      std::this_thread::yield();
      continue;
    }

    IteratorLock.LockHighPri();

    int parallel_job_num = 0;

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

      parallel_job_num = std::min((int)chosen_groups_.size(), 4);
      for (size_t i = 0; i < chosen_groups_.size(); ++i) {
        auto job = compaction_jobs_[i];
        job->group_ = chosen_groups_[i];
        chosen_jobs_.push_back(job);
      }
    }

    auto start_time = GetStartTime();

    std::vector<std::vector<SingleCompactionJob*>> merge_jobs(parallel_job_num);
    int cur_merge = 0;
    for (auto chosen_job : chosen_jobs_) {
      merge_jobs[cur_merge].push_back(chosen_job);
      cur_merge = (cur_merge + 1) % parallel_job_num;
    }

    SingleCompactionJob::thread_pool->SetJobCount(parallel_job_num);
    for (auto& merge_job : merge_jobs) {
      auto func = std::bind(&Compactor::CompactionPreprocess, this, merge_job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    auto preprocess_done_time = GetStartTime();
    float preprocess_time = (preprocess_done_time - start_time) * 1e-6;

    // Sort jobs in each group by key order
    for (auto& mege_job : merge_jobs) {
      std::sort(mege_job.begin(), mege_job.end(), [](SingleCompactionJob* job1, SingleCompactionJob* job2){
        return job1->kv_slices[0].first < job2->kv_slices[0].first;
      });
    }

    db_impl_->SyncCallFlush(merge_jobs);

    auto flush_done_time = GetStartTime();
    float flush_time = (flush_done_time - preprocess_done_time) * 1e-6;

    uint64_t total_out_size = 0;
    for (auto job : chosen_jobs_) {
      total_out_size += job->out_file_size;
    }

    SingleCompactionJob::thread_pool->SetJobCount(parallel_job_num);
    for (auto& merge_job : merge_jobs) {
      auto func = std::bind(&Compactor::CompactionPostprocess, this, merge_job);
      SingleCompactionJob::thread_pool->SubmitJob(func);
    }
    SingleCompactionJob::thread_pool->Join();

    for (auto job : chosen_jobs_) {
      for (size_t i = 0; i < all_compacted_indexes_.size(); ++i) {
        all_compacted_indexes_[i].insert(all_compacted_indexes_[i].end(),
                                         job->compacted_indexes[i].begin(),
                                         job->compacted_indexes[i].end());
        job->compacted_indexes[i].clear();
      }
    }
    vlog_manager_->UpdateBitmap(all_compacted_indexes_);

    for (auto job : chosen_jobs_) {
      group_manager_->AddOperation(job->group_, kOperatorMove, true);
    }

    auto end_time = GetStartTime();
    float postprocess_time = (end_time - flush_done_time) * 1e-6;

    // Use sfence after all vlog are modified
    NVM_BARRIER;

    RECORD_INFO("Flush to base: %.2fMB, %.2fMB, %.3lfs, %.3lf\n",
                total_out_size / 1048576.0, total_out_size / 1048576.0,
                (end_time - start_time) * 1e-6, start_time * 1e-6);

    auto squeezed_in_compaction =
        (float)SqueezedSizeInCompaction.load(std::memory_order_relaxed) / 1048576.0f;
    auto mem_total_size =
        (float)MemTotalSize.load(std::memory_order_relaxed) / 1048576.0f;
    auto compacted_size =
        (float)CompactedSize.load(std::memory_order_relaxed) / 1048576.0f;
    auto total_squeezed_size =
        (float)SqueezedSize.load(std::memory_order_relaxed) / 1048576.0f;
    auto rewrite_count = total_rewrite.load(std::memory_order_relaxed);
    total_rewrite.store(0, std::memory_order_relaxed);

    RECORD_DEBUG("free pages: "
        "[%zu, %zu], size: [%.2fM, %.2fM, %.2fM]; group squeezed %.2fM, rewrite: %d, "
        "time: [%.2fs, %.2fs, %.2fs]\n",
        GetNodeAllocator()->GetNumFreePages(),
        vlog_manager_->free_segments_.size(),
        mem_total_size, compacted_size, total_squeezed_size,
        squeezed_in_compaction, rewrite_count,
        preprocess_time, flush_time, postprocess_time);

    SqueezedSizeInCompaction.store(0);

    // Wait all reading in backup nvm node finish,
    // then we can free vlog pages and nodes.
    while (BackupRead.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    IteratorLock.UnlockHighPri();
  }
}

void Compactor::Reset() {
  StopThread();

  while (BackupRead.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  thread_local uint8_t fingerprints[224] = {0};
  thread_local uint64_t nvm_data[448] = {0};
#endif

  std::unique_lock<std::mutex> lock{mutex_};
  group_manager_->AddOperation(nullptr, kOperationFlushAll, false, this);
  cond_var_.wait(lock);

  std::vector<std::vector<HeatGroup*>> compaction_groups;
  auto cur = chosen_groups_[0];
  while (cur) {
    std::vector<HeatGroup*> compaction_group;
    int size_in_job = 0;
    while (size_in_job < HeatGroup::group_min_size_ && cur) {
      size_in_job += cur->group_size_.load(std::memory_order_relaxed);
      compaction_group.push_back(cur);
      cur = cur->next_seq;
    }

    compaction_groups.push_back(compaction_group);
  }

  int idx = 0;
  for (auto& compaction_group : compaction_groups) {
    SingleCompactionJob* job = compaction_jobs_[idx];
    job->Reset();

    for (auto& heat_group : compaction_group) {
      auto node = heat_group->first_node_->next_node;
      while (node != heat_group->last_node_->next_node) {
        if (NOT_LEAF(node) || !node->heat_group_) {
          node = node->next_node;
          continue;
        }

        job->oldest_key_time_ =
            std::min(job->oldest_key_time_, node->oldest_key_time_);
        MEMCPY(node->nvm_node_->temp_buffer, node->buffer_,
               SIZE_TO_BYTES(GET_NODE_BUFFER_SIZE(node->status_)),
               PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

        std::vector<CompactionRec> read_records(241);

        auto nvm_node = node->nvm_node_;
        int data_size = GET_SIZE(nvm_node->meta.header);
        auto data = nvm_node->data;

        KVStruct tmp_struct{};
        ValueType type;
        SequenceNumber seq_num;

        std::vector<std::string>& keys = job->keys_in_node;

        size_t count = 0;
        for (int i = -16; i < data_size; ++i) {
          tmp_struct.vptr = data[i * 2 + 1];
          if (!tmp_struct.actual_vptr) {
            continue;
          }

          auto& record = read_records[count];
          type = job->vlog_manager_->GetKeyValue(
              tmp_struct.actual_vptr, keys[count],
              record.value_slice, seq_num, record.record_index);
          record.seq_num = (seq_num << 8) | type;
          record.key = &keys[count++];
          record.s = KVStruct{data[i * 2], data[i * 2 + 1]};
        }

        std::stable_sort(read_records.begin(), read_records.begin() + count);

        keys[count] = "";
        read_records[count].key = &keys[count];
        for (size_t i = 1; i <= count; ++i) {
          auto& record = read_records[i];
          auto& last_record = read_records[i - 1];
          if (*record.key != *last_record.key) {
            PutFixed64(last_record.key, last_record.seq_num);
            job->kv_slices.emplace_back(*last_record.key, last_record.value_slice);
          }
        }

        node = node->next_node;
      }
    }

    // TODO: change logic
    if (++idx == num_parallel_compaction_) {
      auto start_time = GetStartTime();
      std::vector<std::vector<SingleCompactionJob*>> temp{compaction_jobs_};
      db_impl_->SyncCallFlush(temp);
      auto end_time = GetStartTime();

      idx = 0;

      uint64_t total_out_size = 0;
      for (auto t : compaction_jobs_) {
        total_out_size += t->out_file_size;
      }

      RECORD_INFO("Flush to base: %.2fMB, %.2fMB, %.3lfs, %.3lf\n",
                  total_out_size / 1048576.0, total_out_size / 1048576.0,
                  (end_time - start_time) * 1e-6, start_time * 1e-6);
    }
  }

  if (idx > 0) {
    std::vector<SingleCompactionJob*> last_jobs;
    for (int i = 0; i < idx; ++i) {
      last_jobs.push_back(compaction_jobs_[i]);
    }

    auto start_time = GetStartTime();
    std::vector<std::vector<SingleCompactionJob*>> temp{last_jobs};
    db_impl_->SyncCallFlush(temp);
    auto end_time = GetStartTime();

    uint64_t total_out_size = 0;
    for (auto t : last_jobs) {
      total_out_size += t->out_file_size;
    }

    RECORD_INFO("Flush to base: %.2fMB, %.2fMB, %.3lfs, %.3lf\n",
                total_out_size / 1048576.0, total_out_size / 1048576.0,
                (end_time - start_time) * 1e-6, start_time * 1e-6);
  }

  MemTotalSize.store(0);
  SqueezedSize.store(0);
  CompactedSize.store(0);
  SqueezedSizeInCompaction.store(0);
  chosen_jobs_.clear();
  for (auto job : compaction_jobs_) {
    job->Reset();
  }

  StartThread();
}

void ReadData(InnerNode* node, SingleCompactionJob* job,
              int& compacted_size, int rewrite_threshold,
              std::vector<KVStruct>& rewrite_kv) {
  auto nvm_node = node->backup_nvm_node_;
  auto new_nvm_node = node->nvm_node_;
  int data_size = GET_SIZE(nvm_node->meta.header);
  size_t parent_level = GET_LEVEL(nvm_node->meta.header) - 1; // level of parent
  auto data = nvm_node->data;

#ifndef ROCKSDB_SUPPORT_THREAD_LOCAL
  uint8_t fingerprints[224] = {0};
  uint64_t nvm_data[448] = {0};
#endif

  KVStruct tmp_struct{};
  ValueType type;
  SequenceNumber seq_num;
  int32_t cur_size = 0;

  std::vector<std::string>& keys = job->keys_in_node;
  std::vector<CompactionRec> read_records(241);

  size_t count = 0;
  for (int i = -16; i < data_size; ++i) {
    tmp_struct.vptr = data[i * 2 + 1];
    if (!tmp_struct.actual_vptr) {
      continue;
    }

    auto& record = read_records[count];
    type = job->vlog_manager_->GetKeyValue(
        tmp_struct.actual_vptr, keys[count],
        record.value_slice, seq_num, record.record_index);
    record.seq_num = (seq_num << 8) | type;
    record.key = &(keys[count]);
    record.s = KVStruct{data[i * 2], data[i * 2 + 1]};
    ++count;
  }

  if (count == 0) {
    return;
  }

  std::stable_sort(read_records.begin(), read_records.begin() + count);

  keys[count] = "";
  read_records[count].key = &keys[count];
  int insert_times = 0;

  size_t rewrite_count = 0;
  for (size_t i = 1; i <= count; ++i) {
    auto& record = read_records[i];
    auto& last_record = read_records[i - 1];
    insert_times += last_record.s.insert_times;

    if (*record.key == *last_record.key) {
      job->compacted_indexes[last_record.s.actual_vptr >> 20]
          .push_back(last_record.record_index);
    } else {
      if (insert_times > rewrite_threshold) {
        last_record.s.insert_times = 1;
        cur_size += last_record.s.kv_size;
        nvm_data[rewrite_count * 2] = last_record.s.hash;
        nvm_data[rewrite_count * 2 + 1] = last_record.s.vptr;
        fingerprints[rewrite_count++] = static_cast<uint8_t>(last_record.s.actual_hash);

        int bucket = (int)(last_record.s.actual_hash & 63);
        int h = (int)(last_record.s.actual_hash >> 6);
        uint8_t digit = h == 0 ? 0 : __builtin_ctz(h) + 1;
        node->hll_[bucket] = std::max(node->hll_[bucket], digit);

        Rehash(last_record.s, *last_record.key, parent_level);
        rewrite_kv.push_back(last_record.s);
      } else {
        job->compacted_indexes[last_record.s.actual_vptr >> 20]
            .push_back(last_record.record_index);
        PutFixed64(last_record.key, last_record.seq_num);
        job->kv_slices.emplace_back(*last_record.key, last_record.value_slice);
      }
      insert_times = 0;
    }
  }

  int flush_size = ALIGN_UP(rewrite_count, 16);
  int flush_rows = SIZE_TO_ROWS(flush_size);
  assert(flush_rows <= NVM_MAX_ROWS);
  assert(flush_size <= NVM_MAX_SIZE);

  memset(nvm_data + (rewrite_count * 2), 0,
         SIZE_TO_BYTES(flush_size - rewrite_count));
  memset(fingerprints + rewrite_count, 0, flush_size - rewrite_count);
  MEMCPY(new_nvm_node->data, nvm_data, SIZE_TO_BYTES(flush_size),
         PMEM_F_MEM_NONTEMPORAL);
  NVM_BARRIER;
  MEMCPY(new_nvm_node->meta.fingerprints_, fingerprints, flush_size,
         PMEM_F_MEM_NODRAIN | PMEM_F_MEM_NONTEMPORAL);

  uint64_t hdr = new_nvm_node->meta.header;
  SET_SIZE(hdr, flush_size);
  SET_ROWS(hdr, flush_rows);
  new_nvm_node->meta.header = hdr;
  PERSIST(new_nvm_node, CACHE_LINE_SIZE);

  node->estimated_size_ = cur_size;
  compacted_size -= cur_size;
}

void Compactor::CompactionPreprocess(std::vector<SingleCompactionJob*> jobs) {
  for (auto job : jobs) {
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

    std::vector<KVStruct> rewrite_kv;
    rewrite_kv.reserve(256);

    InnerNode* cur_node = start_node->next_node;
    while (cur_node != next_start_node) {
      cur_node->opt_lock_.lock();
      cur_node->share_mutex_.lock();

      if (NOT_LEAF(cur_node) || !cur_node->heat_group_) {
        cur_node->share_mutex_.unlock();
        cur_node->opt_lock_.unlock();
        cur_node = cur_node->next_node;
        continue;
      }

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

      if (cur_parent && cur_node->parent_node != cur_parent) {
        ProcessNodes(cur_parent, children, job, rewrite_kv);
        rewrite_kv.clear();
        children.clear();
      }

      ReadData(cur_node, job, compacted_size, rewrite_threshold_, rewrite_kv);
      cur_node->share_mutex_.unlock();

      children.push_back(cur_node);
      cur_parent = cur_node->parent_node;
      cur_node = cur_node->next_node;
    }

    if (cur_parent && !children.empty()) {
      ProcessNodes(cur_parent, children, job, rewrite_kv);
    }

    job->candidates.push_back(nullptr);
    job->group_->group_size_.fetch_add(-compacted_size, std::memory_order_relaxed);
    UpdateTotalSize(-compacted_size);
    SqueezedSizeInCompaction.fetch_add(squeezed_size, std::memory_order_release);
    CompactedSize.fetch_add(compacted_size, std::memory_order_release);
  }
}

void Compactor::CompactionPostprocess(std::vector<SingleCompactionJob*> jobs) {
  for (auto& job : jobs) {
    // remove compacted node form list
    auto cur_node = job->group_->first_node_;
    auto node_after_end = job->group_->last_node_->next_node;
    auto candidates = job->candidates;
    assert(IS_GROUP_START(cur_node));

    auto candidate = candidates.front();
    candidates.pop_front();
    while (candidate && cur_node != node_after_end) {
      std::lock_guard<RWSpinLock> link_lk(cur_node->link_lock_);
      if (cur_node->next_node == candidate) {
        RemoveOldNVMNode(cur_node);
        candidate = candidates.front();
        candidates.pop_front();
      }
      cur_node = cur_node->next_node;
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

    job->group_->status_.store(kGroupWaitMove, std::memory_order_relaxed);
  }
}

void TimerCompaction::SetDB(DBImpl* db_impl) {
  this->db_impl_ = db_impl;
}

void TimerCompaction::StopCompaction() {
  this->flag_.store(false);
}

void TimerCompaction::BGWork() {
  while (flag_.load()) {
    sleep(interval_);
    db_impl_->TryScheduleCompaction();
  }
}

} // namespace ROCKSDB_NAMESPACE