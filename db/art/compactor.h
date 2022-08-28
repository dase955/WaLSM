//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <deque>
#include <db/art/utils.h>
#include <condition_variable>
#include <db/dbformat.h>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/threadpool.h>
#include <rocksdb/threadpool.h>
#include <util/autovector.h>

namespace ROCKSDB_NAMESPACE {

struct NVMNode;
struct ArtNode;
struct HeatGroup;
struct InnerNode;
class DBImpl;
class VLogManager;
class GlobalMemtable;
class HeatGroupManager;

struct SingleCompactionJob {
  static ThreadPool* thread_pool;

  HeatGroup*   group_;
  VLogManager* vlog_manager_;
  uint64_t     out_file_size;
  int64_t      oldest_key_time_;

  std::deque<InnerNode*>   candidates;
  std::vector<InnerNode*>  candidates_removed;
  std::vector<InnerNode*>  candidate_parents;
  std::vector<ArtNode*>    removed_arts;
  std::vector<std::pair<std::string, Slice>>   kv_slices;

  std::vector<std::string> keys_in_node;
  autovector<RecordIndex>* compacted_indexes;

  void Reset() {
    candidates.clear();
    candidates_removed.clear();
    candidate_parents.clear();
    removed_arts.clear();
    kv_slices.clear();
  }
};

class Compactor : public BackgroundThread {
 public:
  explicit Compactor(const DBOptions& options);

  ~Compactor() noexcept override;

  void Reset();

  void SetGroupManager(HeatGroupManager* group_manager);

  void SetVLogManager(VLogManager* vlog_manager);

  void SetDB(DBImpl* db_impl);

  void Notify(std::vector<HeatGroup*>& heat_groups);

  int GetNumParallelCompaction() const {
    return num_parallel_compaction_;
  }

  static int64_t compaction_threshold_;

 private:
  void BGWork() override;

  void CompactionPreprocess(SingleCompactionJob* job);

  void CompactionPostprocess(SingleCompactionJob* job);

  HeatGroupManager* group_manager_ = nullptr;

  VLogManager* vlog_manager_ = nullptr;

  DBImpl* db_impl_ = nullptr;

  int num_parallel_compaction_;

  int rewrite_threshold_;

  std::vector<HeatGroup*> chosen_groups_;

  std::vector<SingleCompactionJob*> chosen_jobs_;

  std::vector<SingleCompactionJob*> compaction_jobs_;

  std::vector<std::vector<RecordIndex>> all_compacted_indexes_;
};

int64_t GetMemTotalSize();

void UpdateTotalSize(int32_t update_size);

void UpdateTotalSqueezedSize(int64_t update_size);

// record number of read in backup nvm node
void IncrementBackupRead();

void ReduceBackupRead();

// Before creating db iterator, we must notify compactor that
// we want to do scan operation. Because we can't do compaction when doing scan.
void TryCreateIterator();

void DeleteIterator();

} // namespace ROCKSDB_NAMESPACE