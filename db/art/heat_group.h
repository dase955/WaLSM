//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <mutex>
#include <atomic>
#include <vector>
#include <sstream>
#include <iomanip>
#include <condition_variable>
#include <rocksdb/rocksdb_namespace.h>
#include "macros.h"
#include "timestamp.h"

namespace ROCKSDB_NAMESPACE {

struct InnerNode;
struct HeatGroup;
class HeatGroupManager;

enum GroupStatus {
  kGroupNone,
  kGroupWaitMove,
  kGroupWaitSplit,
  kGroupCompaction,
};

enum GroupOperator {
  kOperatorMove,
  kOperatorSplit,
  kOperatorMerge,
  kOperatorLevelDown,
  kOperationChooseCompaction
};

struct GroupOperation {
  HeatGroup* target;
  GroupOperator op;
  void* arg;

  GroupOperation() = default;

  GroupOperation(HeatGroup* target_, GroupOperator op_, void* arg_)
      : target(target_), op(op_), arg(arg_) {};
};

struct HeatGroup {
  std::mutex lock;
  char padding[24];

  Timestamps               ts;

  std::atomic<int32_t>     group_size_;
  InnerNode*               first_node_;
  InnerNode*               last_node_;
  std::atomic<GroupStatus> status_;

  bool in_base_layer; // Groups smaller than threshold are in base layer
  bool in_temp_layer;
  bool is_removed;    // Removed group will not be used anymore

  // next_seq and prev_seq present physically adjacent node,
  // used for group merge
  HeatGroup*               next_seq = nullptr;
  HeatGroup*               prev_seq = nullptr;

  // next and prev present adjacent node in heat group queue
  HeatGroupManager*        group_manager_;
  HeatGroup*               next;
  HeatGroup*               prev;

  static int group_min_size_;

  static int group_split_threshold_;

  static int force_decay_waterline_;

  static int layer_ts_interval_;

  static float decay_factor_[32];

  explicit HeatGroup(InnerNode* initial_node = nullptr);

  bool InsertNewNode(InnerNode* last, std::vector<InnerNode*>& inserts);

  void UpdateSize(int32_t size);

  void UpdateSqueezedSize(int32_t size);

  void UpdateHeat();

  void MaybeScheduleHeatDecay(int32_t last_ts);
};

struct MultiLayerGroupQueue {
  HeatGroup*  heads_[MAX_LAYERS + 2]{};
  HeatGroup*  tails_[MAX_LAYERS + 2]{};
  HeatGroup** heads = heads_ + 2;
  HeatGroup** tails = tails_ + 2;
  int         total_layers = 1;

  void CountGroups() const {
    std::stringstream ss;
    for (int l = -2; l < MAX_LAYERS; ++l) {
      int count = 0;
      int64_t sum = 0;
      auto cur = heads[l]->next;
      while (cur != tails[l]) {
        sum += cur->group_size_.load(std::memory_order_relaxed);
        cur = cur->next;
        ++count;
      }
      sum /= 1048576;
      ss << std::setw(4) << count << "(" << sum << "M) ";
    }
    printf("Group counts: %s\n", ss.str().c_str());
  }
};

inline float CalculateHeat(int32_t ts) {
  static float base[32] = {
      1.0, 1.021897, 1.0442734786090002, 1.0671399349701014,
      1.0905070981261418, 1.11438593205381, 1.1387876408079924, 1.163723673778765,
      1.1892057310634987, 1.215245768956596, 1.2418560055594388, 1.269048926513174,
      1.296837290857033, 1.3252341370149294, 1.3542527889131455, 1.3839068622319768,
      1.4142102707942703, 1.4451772330938526, 1.4768222789669088, 1.5091602564094473,
      1.542206338544045, 1.575976030739144, 1.6104851778842393, 1.6457499718243704,
      1.6817869589574088, 1.7186130479976993, 1.756245517909705, 1.7947020260153739,
      1.8340006162790325, 1.8741597277736948, 1.9151982033327555, 1.9571352983911328
  };

  static float multiplier[64] = {
      1.52587890625e-05, 3.0517578125e-05, 6.103515625e-05, 0.0001220703125,
      0.000244140625, 0.00048828125, 0.0009765625, 0.001953125,
      0.00390625, 0.0078125, 0.015625, 0.03125,
      0.0625, 0.125, 0.25, 0.5,

      1.0, 2.0, 4.0, 8.0,
      16.0, 32.0, 64.0, 128.0,
      256.0, 512.0, 1024.0, 2048.0,
      4096.0, 8192.0, 16384.0, 32768.0,
      65536.0, 131072.0, 262144.0, 524288.0,
      1048576.0, 2097152.0, 4194304.0, 8388608.0,
      16777216.0, 33554432.0, 67108864.0, 134217728.0,
      268435456.0, 536870912.0, 1073741824.0, 2147483648.0,
      4294967296.0, 8589934592.0, 17179869184.0, 34359738368.0,
      68719476736.0, 137438953472.0, 274877906944.0, 549755813888.0,
      1099511627776.0, 2199023255552.0, 4398046511104.0, 8796093022208.0,
      17592186044416.0, 35184372088832.0, 70368744177664.0, 140737488355328.0

  };

  ts += 512;
  assert(ts < 2048);
  return ts >= 0 ? base[ts & 31] * multiplier[ts >> 5] : 0.0f;
}

inline float GetDecayFactor(int delta) {
  delta /= HeatGroup::layer_ts_interval_;
  return delta < 32 ? HeatGroup::decay_factor_[delta] : 0;
}

// Choose new level of heat group by estimated upper and lower bound.
int ChooseGroupLevel(HeatGroup* group);

// Insert new nodes to heat group of previous node.
void InsertNodesToGroup(InnerNode* node, InnerNode* inserts);

void InsertNodesToGroup(InnerNode* node, std::vector<InnerNode*>& inserts);

} // namespace ROCKSDB_NAMESPACE