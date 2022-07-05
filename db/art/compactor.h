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
  std::vector<uint64_t>    rewrite_data;

  std::vector<std::string> keys_in_node;
  autovector<RecordIndex>* compacted_indexes;
  std::vector<std::pair<NVMNode*, int>> nvm_nodes_and_sizes;

  void Reset() {
    rewrite_data.clear();
    candidates.clear();
    candidates_removed.clear();
    candidate_parents.clear();
    removed_arts.clear();
    nvm_nodes_and_sizes.clear();
  }
};

class Compactor : public BackgroundThread {
 public:
  explicit Compactor(const DBOptions& options);

  ~Compactor() noexcept override;

  void Reset();

  void SetGroupManager(HeatGroupManager* group_manager);

  void SetVLogManager(VLogManager* vlog_manager);

  void SetGlobalMemtable(GlobalMemtable* global_memtable);

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

  void RewriteData(SingleCompactionJob* job);

  GlobalMemtable* global_memtable_ = nullptr;

  HeatGroupManager* group_manager_ = nullptr;

  VLogManager* vlog_manager_ = nullptr;

  DBImpl* db_impl_ = nullptr;

  int num_parallel_compaction_;

  std::vector<HeatGroup*> chosen_groups_;

  std::vector<SingleCompactionJob*> chosen_jobs_;

  std::vector<SingleCompactionJob*> compaction_jobs_;
};

int64_t GetMemTotalSize();

void UpdateTotalSize(int32_t update_size);

void UpdateTotalSqueezedSize(int64_t update_size);

// record number of read in backup nvm node
void IncrementBackupRead();

void ReduceBackupRead();

} // namespace ROCKSDB_NAMESPACE