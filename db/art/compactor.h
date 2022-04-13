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

struct CompactionRec {
  std::string key;
  std::string value;
  uint64_t seq_num_;
  ValueType   type;

  CompactionRec() = default;

  CompactionRec(std::string& key_, std::string& value_, uint64_t seq_num, ValueType type_)
      : key(key_), value(value_), seq_num_(seq_num), type(type_){};

  friend bool operator<(const CompactionRec& l, const CompactionRec& r) {
    return l.key < r.key;
  }
};

struct ArtCompactionJob {
  uint64_t total_num_entries_ = 0;
  uint64_t total_num_deletes_ = 0;
  uint64_t total_data_size_ = 0;
  size_t total_memory_usage_ = 0;
  uint64_t oldest_key_time_;
  std::vector<CompactionRec> compacted_data_;
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