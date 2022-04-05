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
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

struct HeatGroup;
class HeatGroupManager;
class VLogManager;
struct NVMNode;

class Compactor {
 public:
  Compactor() : thread_stop_(false), chosen_group_(nullptr),
                group_manager_(nullptr), vlog_manager_(nullptr) {};

  void SetGroupManager(HeatGroupManager *group_manager);

  void SetVLogManager(VLogManager *vlog_manager);

  void BGWorkDoCompaction();

  void StopBGWork();

  void Notify(HeatGroup *heat_group);

 private:
  // return compacted size
  int32_t DoCompaction();

  void ReadAndSortNodeData(NVMNode *nvmNode);

  std::mutex mutex_;

  std::condition_variable cond_var_;

  bool thread_stop_;

  HeatGroup *chosen_group_;

  HeatGroupManager *group_manager_;

  VLogManager *vlog_manager_;
};

void UpdateTotalSize(size_t update_size);

} // namespace ROCKSDB_NAMESPACE