//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <mutex>
#include <condition_variable>
#include <rocksdb/rocksdb_namespace.h>
#include "utils.h"
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

struct InnerNode;

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

struct HeatGroup {
  std::mutex lock; // TODO: check sizeof(std::mutex>
  char padding[24];

  std::atomic<int32_t>     group_size_;
  InnerNode                *first_node_;
  InnerNode                *last_node_;
  std::atomic<GroupStatus> status_;
  bool                     in_base_layer_;

  HeatGroup                *next;
  HeatGroup                *prev;

  explicit HeatGroup(InnerNode *initial_node = nullptr)
      : group_size_(0),
        first_node_(initial_node), last_node_(initial_node),
        status_(kGroupNone), in_base_layer_(false),
        next(nullptr), prev(nullptr) {};

  bool insertNewNode(InnerNode *last, std::vector<InnerNode*> &inserts);

  void updateSize(int32_t size);
};

HeatGroup *NewHeatGroup();

struct GroupOperation {
  HeatGroup *target;
  GroupOperator op;

  GroupOperation() = default;
  GroupOperation(HeatGroup *target_, GroupOperator op_)
      : target(target_), op(op_) {};
};

struct MultiLayerGroupQueue {
  HeatGroup  *heads_[MAX_LAYERS + 2];
  HeatGroup  *tails_[MAX_LAYERS + 2];
  HeatGroup **heads = heads_ + 2;
  HeatGroup **tails = tails_ + 2;
  int totalLayers = 1;
};

void InitGroupQueue();

void RemoveFromQueue(HeatGroup *group);

void InsertIntoLayer(HeatGroup *inserted, int level);

void GroupInsertNewNode(InnerNode *node, InnerNode* inserts);

void GroupInsertNewNode(InnerNode *node, std::vector<InnerNode*> &inserts);

int ChooseGroupLevel(HeatGroup *group);

void AddOperation(HeatGroup *group, GroupOperator op, bool highPri = false);

float CalculateHeat(int32_t ts);

void EstimateLowerAndUpperBound(Timestamps &ts, float &lower_bound, float &upper_bound);

int32_t GetGlobalDec();

void HeatGroupLoop();

void WaitHeatGroupLoopStop();

} // namespace ROCKSDB_NAMESPACE