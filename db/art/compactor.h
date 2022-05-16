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
#include <condition_variable>
#include <db/dbformat.h>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb//threadpool.h>
#include <util/autovector.h>

namespace ROCKSDB_NAMESPACE {

struct HeatGroup;
class HeatGroupManager;
class VLogManager;
struct InnerNode;
struct NVMNode;
class DBImpl;

struct SingleCompactionJob {
  static ThreadPool* thread_pool;

  HeatGroup* group_;
  int64_t oldest_key_time_;
  InnerNode* start_node_;
  InnerNode* node_after_end_;
  VLogManager* vlog_manager_;
  autovector<RecordIndex>* compacted_indexes_;
  std::deque<InnerNode*> candidates_;
  std::vector<std::string> keys_in_node;
  std::vector<std::pair<NVMNode*, int>> nvm_nodes_and_sizes;
};

class Compactor {
 public:
  Compactor() : thread_stop_(false), group_manager_(nullptr),
                vlog_manager_(nullptr) {};

  ~Compactor() noexcept;

  void SetGroupManager(HeatGroupManager* group_manager);

  void SetVLogManager(VLogManager* vlog_manager);

  void SetDB(DBImpl* db_impl);

  void StartCompactionThread();

  void StopCompactionThread();

  void BGWorkDoCompaction();

  void Notify(std::vector<HeatGroup*>& heat_groups);

  static int64_t compaction_threshold_;

 private:
  void CompactionPreprocess(SingleCompactionJob* job);

  void CompactionPostprocess(SingleCompactionJob* job);

  std::mutex mutex_;

  std::condition_variable cond_var_;

  std::thread compactor_thread_;

  bool thread_stop_;

  HeatGroupManager* group_manager_;

  VLogManager* vlog_manager_;

  DBImpl* db_impl_;

  int num_parallel_compaction_ = 4;

  std::vector<HeatGroup*> chosen_groups_;

  std::vector<SingleCompactionJob*> chosen_jobs_;

  std::vector<SingleCompactionJob*> compaction_jobs_;
};

int64_t GetMemTotalSize();

void UpdateTotalSize(int32_t update_size);

// record number of read in backup nvm node
void IncrementBackupRead();

void ReduceBackupRead();

} // namespace ROCKSDB_NAMESPACE