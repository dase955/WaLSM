//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <db/dbformat.h>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

struct HeatGroup;
class HeatGroupManager;
class VLogManager;
struct NVMNode;
class DBImpl;

struct ArtCompactionJob {
  /*uint64_t total_num_entries_ = 0;
  uint64_t total_num_deletes_ = 0;
  uint64_t total_data_size_ = 0;
  size_t total_memory_usage_ = 0;*/
  uint64_t oldest_key_time_;

  VLogManager* vlog_manager_;
  std::vector<NVMNode*> nvm_nodes_;
  std::vector<RecordIndex>* compacted_indexes_;
};

class Compactor {
 public:
  Compactor() : thread_stop_(false), chosen_group_(nullptr),
                group_manager_(nullptr), vlog_manager_(nullptr) {};

  void SetGroupManager(HeatGroupManager* group_manager);

  void SetVLogManager(VLogManager* vlog_manager);

  void SetDB(DBImpl* db_impl);

  void StartCompactionThread();

  void BGWorkDoCompaction();

  void StopCompactionThread();

  void Notify(HeatGroup* heat_group);

  // return compacted size
  void DoCompaction();

  static size_t compaction_threshold_;

 private:

  std::mutex mutex_;

  std::condition_variable cond_var_;

  std::thread compactor_thread_;

  bool thread_stop_;

  HeatGroup* chosen_group_;

  HeatGroupManager* group_manager_;

  VLogManager* vlog_manager_;

  DBImpl* db_impl_;
};

void UpdateTotalSize(size_t update_size);

} // namespace ROCKSDB_NAMESPACE