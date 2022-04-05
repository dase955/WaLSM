//
// Created by joechen on 2022/4/5.
//

#pragma once
#include <rocksdb/rocksdb_namespace.h>
#include <thread>
#include <atomic>
#include "heat_group.h"
#include "concurrent_queue.h"

namespace ROCKSDB_NAMESPACE {

extern std::atomic<int32_t> GlobalDecay;

class Compactor;

class HeatGroupManager {
 public:
  void SetCompactor(Compactor *compactor);

  void InitGroupQueue();

  void StartHeatThread();

  void StopHeatThread();

  // Operation with high priority will be put in front
  void AddOperation(HeatGroup *group, GroupOperator op, bool highPri = false);

  void InsertIntoLayer(HeatGroup *inserted, int level);

  void MoveGroup(HeatGroup *group);

  void MoveAllGroupsToLayer(int from, int to);

 private:
  void BGWorkProcessHeatGroup();

  void ChooseCompaction();

  // Different from GroupLevelDown,
  // groups in layer1 will be moved to layer0 even if layer0 is not empty
  void ForceGroupLevelDown();

  void GroupLevelDown();

  Compactor *compactor_;

  TQueueConcurrent<GroupOperation> group_operations_;

  MultiLayerGroupQueue group_queue_;

  std::thread heat_processor_;
};

} // namespace ROCKSDB_NAMESPACE
