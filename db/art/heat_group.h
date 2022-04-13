//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <mutex>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <rocksdb/rocksdb_namespace.h>
#include "macros.h"
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

struct InnerNode;
struct HeatGroup;
class HeatGroupManager;

// Status of heat group
enum GroupStatus {
  kGroupNone,
  kGroupWaitSplit,
  kGroupWaitMove,
  kGroupCompaction,
};

enum GroupOperator {
  kOperatorSplit,
  kOperatorMove,
  kOperatorLevelDown,
  kOperationChooseCompaction,
  kOperationStop,
};

struct GroupOperation {
  HeatGroup* target;
  GroupOperator op;

  GroupOperation() = default;
  GroupOperation(HeatGroup* target_, GroupOperator op_)
      : target(target_), op(op_) {};
};

struct HeatGroup {
  std::mutex lock;
  char padding[24];

  Timestamps               ts;

  std::atomic<int32_t>     group_size_;
  InnerNode*               first_node_;
  InnerNode*               last_node_;
  std::atomic<GroupStatus> status_;
  bool                     in_base_layer_;

  HeatGroupManager*        group_manager_;
  HeatGroup*               next;
  HeatGroup*               prev;
  char padding2[8];

  static int group_min_size_;

  static int group_split_threshold_;

  static int force_decay_waterline_;

  static int layer_ts_interval_;

  static float decay_factor_[32];

  explicit HeatGroup(InnerNode* initial_node = nullptr);

  bool InsertNewNode(InnerNode* last, std::vector<InnerNode*>& inserts);

  void UpdateSize(int32_t size);

  void UpdateHeat();

  void MaybeScheduleHeatDecay(int32_t last_ts);
};

struct MultiLayerGroupQueue {
  HeatGroup*  heads_[MAX_LAYERS + 2]{};
  HeatGroup*  tails_[MAX_LAYERS + 2]{};
  HeatGroup** heads = heads_ + 2;
  HeatGroup** tails = tails_ + 2;
  int totalLayers = 1;
};

float CalculateHeat(int32_t ts);

inline float GetDecayFactor(int delta) {
  delta /= HeatGroup::layer_ts_interval_;
  return delta < 32 ? HeatGroup::decay_factor_[delta] : 0;
}

// Choose new level of heat group by estimated upper and lower bound.
int ChooseGroupLevel(HeatGroup* group);

// Estimate heat bound by current timestamp.
void EstimateLowerAndUpperBound(
    Timestamps& ts, float& lower_bound, float& upper_bound);

// Insert new nodes to heat group of previous node.
void InsertNodesToGroup(InnerNode* node, InnerNode* inserts);

void InsertNodesToGroup(InnerNode* node, std::vector<InnerNode*>& inserts);

} // namespace ROCKSDB_NAMESPACE