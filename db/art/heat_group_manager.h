//
// Created by joechen on 2022/4/5.
//

#pragma once
#include <rocksdb/rocksdb_namespace.h>
#include <thread>
#include <atomic>
#include "utils.h"
#include "heat_group.h"
#include "concurrent_queue.h"

namespace ROCKSDB_NAMESPACE {

extern std::atomic<int32_t> GlobalDecay;

class Compactor;

class HeatGroupManager : public BackgroundThread {
 public:
  explicit HeatGroupManager(const DBOptions& options);

  void Reset();

  void AddOperation(HeatGroup* group, GroupOperator op,
                    bool high_pri = false, void* arg = nullptr);

  void InsertIntoLayer(HeatGroup* inserted, int level);

  float CountDramUsageMib() const {
    return group_queue_.CountDramUsageMib();
  }

 private:
  void BGWork() override;

  void ChooseCompaction(Compactor* compactor, size_t num_chosen);

  void ChooseFirstGroup(Compactor* compactor);

  bool MergeNextGroup(HeatGroup* group, HeatGroup* next_group);

  void TryMergeBaseLayerGroups();

  void SplitGroup(HeatGroup* group);

  void MoveGroup(HeatGroup* group);

  void ForceGroupLevelDown();

  void GroupLevelDown();

  bool CheckForCompaction(HeatGroup* group, int cur_level);

  void MoveAllGroupsToLayer(int from, int to);

  TQueueConcurrent<GroupOperation> group_operations_;

  MultiLayerGroupQueue group_queue_;
};

} // namespace ROCKSDB_NAMESPACE
