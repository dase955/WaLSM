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
  uint64_t    seqNum;
  ValueType   type;

  CompactionRec() = default;

  CompactionRec(std::string &key_, std::string &value_, uint64_t seqNum_, ValueType type_)
      : key(key_), value(value_), seqNum(seqNum_), type(type_){};

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

  void SetGroupManager(HeatGroupManager *group_manager);

  void SetVLogManager(VLogManager *vlog_manager);

  void SetDB(DBImpl *db_impl);

  void BGWorkDoCompaction();

  void StopBGWork();

  void Notify(HeatGroup *heat_group);

 private:
  // return compacted size
  int32_t DoCompaction();

  std::mutex mutex_;

  std::condition_variable cond_var_;

  bool thread_stop_;

  HeatGroup *chosen_group_;

  HeatGroupManager *group_manager_;

  VLogManager *vlog_manager_;

  DBImpl *db_impl_;
};

void UpdateTotalSize(size_t update_size);

} // namespace ROCKSDB_NAMESPACE