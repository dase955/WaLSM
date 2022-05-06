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
  HeatGroupManager(const DBOptions& options);

  ~HeatGroupManager();

  void SetCompactor(Compactor* compactor);

  void InitGroupQueue(float coeff);

  // Operation with high priority will be put in front
  void AddOperation(HeatGroup* group, GroupOperator op, bool highPri = false);

  void InsertIntoLayer(HeatGroup* inserted, int level);

  void TestChooseCompaction();

 private:
  void BGWorkProcessHeatGroup();

  void ChooseCompaction(size_t num);

  // Split heat group
  void SplitGroup(HeatGroup* group);

  void MoveGroup(HeatGroup* group);

  // Different from GroupLevelDown,
  // groups in layer1 will be moved to layer0 even if layer0 is not empty
  void ForceGroupLevelDown();

  void GroupLevelDown();

  void MoveAllGroupsToLayer(int from, int to);

  Compactor* compactor_;

  TQueueConcurrent<GroupOperation> group_operations_;

  MultiLayerGroupQueue group_queue_;

  std::thread heat_processor_;
};

} // namespace ROCKSDB_NAMESPACE
