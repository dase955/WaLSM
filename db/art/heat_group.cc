//
// Created by joechen on 2022/4/3.
//

#include "heat_group.h"
#include "heat_group_manager.h"

#include <mutex>
#include <queue>
#include <cmath>
#include <cassert>
#include <immintrin.h>

#include "utils.h"
#include "timestamp.h"
#include "global_memtable.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

std::atomic<int32_t> GlobalDecay{0};

static float   LayerHeatBound[MAX_LAYERS + 1];

/*----------------------------------------------------------------------*/

bool HeatGroup::InsertNewNode(
    InnerNode* last, std::vector<InnerNode*>& inserts) {
  std::lock_guard<std::mutex> lk(lock);

  // Nodes have been inserted into heat groups, just return
  if (inserts.front()->heat_group_) {
    return true;
  }

  if (last->heat_group_ != this) {
    return false;
  }

  if (last_node_ == last) {
    last_node_ = inserts.back();
  }

  for (auto &inserted : inserts) {
    inserted->heat_group_ = this;
  }

  return true;
}

void HeatGroup::UpdateSize(int32_t size) {
  // This is just a rough estimation, so we don't acquire lock.

  auto cur_status = status_.load(std::memory_order_relaxed);
  auto cur_size = group_size_.fetch_add(size, std::memory_order_relaxed);
  UpdateTotalSize(size);

  if (cur_status != kGroupNone) {
    return;
  }

  GroupStatus new_status = kGroupNone;
  GroupOperator op;

  if (in_base_layer_ && cur_size > GroupMinSize) {
    new_status = kGroupWaitMove;
    op = kOperatorMove;
  } else if (cur_size > GroupSplitThreshold) {
    new_status = kGroupWaitSplit;
    op = kOperatorSplit;
  }

  if (new_status != kGroupNone &&
      status_.compare_exchange_strong(
          cur_status, new_status, std::memory_order_relaxed)) {
    group_manager_->AddOperation(this, op);
  }
}

void HeatGroup::UpdateHeat() {
  ts.UpdateHeat();
  MaybeScheduleHeatDecay(ts.last_ts_);
}

inline void HeatGroup::MaybeScheduleHeatDecay(int32_t last_ts) {
  int32_t decay = GlobalDecay.load(std::memory_order_relaxed);
  if (last_ts - decay > Waterline &&
      GlobalDecay.compare_exchange_strong(
          decay, decay + ForceDecay, std::memory_order_relaxed)) {
    group_manager_->AddOperation(nullptr, kOperatorLevelDown, true);
  }
}

/*----------------------------------------------------------------------*/

float CalculateHeat(int32_t ts) {
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

void EstimateLowerAndUpperBound(
    Timestamps& ts, float& lower_bound, float& upper_bound) {
  int32_t begin_ts, mid_ts, end_ts;
  float heat;

  if (!ts.GetCurrentHeatAndTs(begin_ts, mid_ts, end_ts, heat)) {
    lower_bound = upper_bound = 0.f;
    return;
  }

  float h1 = CalculateHeat(begin_ts);
  float h2 = CalculateHeat(mid_ts);
  float h3 = CalculateHeat(end_ts);
  lower_bound = heat + (h1 + h2) * 4;
  upper_bound = heat + (h2 + h3) * 4;
}

/*----------------------------------------------------------------------*/

void SplitGroup(HeatGroup* group) {
  // This group maybe has been chosen to do compaction, so just pass
  if (group->status_.load(std::memory_order_relaxed) != kGroupWaitSplit) {
    return;
  }

  group->lock.lock();

  auto cur_ts = group->ts.Copy();

  auto left_start = group->first_node_;
  auto right_end = group->last_node_;
  auto left_end = left_start;

  auto total_size = group->group_size_.load(std::memory_order_relaxed);
  auto split_point = total_size >> 1;
  int32_t left_size = 0;

  while (left_end != right_end) {
    left_size += left_end->estimated_size_;
    left_end->heat_group_ = group;
    if (left_size > split_point) {
      break;
    }
    left_end = left_end->next_node_;
  }

  // TODO: fix this corner case
  assert(left_end != right_end);

  group->last_node_ = left_end;
  group->group_size_ = left_size;

  auto right_group = new HeatGroup();
  right_group->lock.lock();
  right_group->ts = cur_ts;
  auto right_start = left_end->next_node_;
  right_group->first_node_ = right_start;
  right_group->last_node_ = right_end;
  right_group->group_size_ = total_size - left_size;
  right_group->group_manager_ = group->group_manager_;
  while (right_start != right_end) {
    right_start->heat_group_ = right_group;
    right_start = right_start->next_node_;
  }

  group->group_manager_->MoveGroup(group);
  group->group_manager_->InsertIntoLayer(right_group, ChooseGroupLevel(right_group));

  right_group->status_.store(kGroupNone, std::memory_order_release);
  group->status_.store(kGroupNone, std::memory_order_release);
  right_group->lock.unlock();
  group->lock.unlock();
}

int ChooseLevelByHeat(float heat) {
#ifdef __amd64__
  __m128 cmp1, cmp2;
  __m128 h = _mm_set1_ps(heat);
  cmp1 = _mm_cmplt_ps( h, _mm_loadu_ps(reinterpret_cast<const float *>(LayerHeatBound + 1)));
  cmp2 = _mm_cmplt_ps( h, _mm_loadu_ps(reinterpret_cast<const float *>(LayerHeatBound + 5)));
  int mask = (1 << 4) - 1;
  int bitfield = ((_mm_movemask_ps(cmp2) & mask) << 4) | (_mm_movemask_ps(cmp1) & mask);
  int level = bitfield == 0 ? 9 :  __builtin_ctz(bitfield);
  return level;
#else
  for (int i = 0; i < 8; i++) {
    if (heat < LayerHeatBound[i + 1]) {
      return i;
    }
  }
  return 8;
#endif
}

int ChooseGroupLevel(HeatGroup* group) {
  if (group->group_size_ < GroupMinSize) {
    return -1;
  }

  float lowerBound = 0.0f;
  float upperBound = 0.0f;

  EstimateLowerAndUpperBound(group->ts, lowerBound, upperBound);
  int lowerLevel = ChooseLevelByHeat(lowerBound);
  int upperLevel = ChooseLevelByHeat(upperBound);
  return (lowerLevel + upperLevel) >> 1;
}

void InsertNodesToGroup(InnerNode* node, InnerNode* insert) {
  std::vector<InnerNode*> inserts{insert};
  InsertNodesToGroup(node, inserts);
}

void InsertNodesToGroup(InnerNode* node, std::vector<InnerNode*>& inserts) {
  for (size_t tries = 0;; ++tries) {
    auto heat_group = node->heat_group_;
    if (heat_group && heat_group->InsertNewNode(node, inserts)) {
      // success
      break;
    }
    port::AsmVolatilePause();
    if (tries > 100) {
      std::this_thread::yield();
    }
  }
}

void RemoveFromQueue(HeatGroup* group) {
  group->prev->next = group->next;
  group->next->prev = group->prev;
}

/*----------------------------------------------------------------------*/

void HeatGroupManager::InitGroupQueue() {
  // Do some pre calculation
  for (size_t i = 0; i < MAX_LAYERS + 2; ++i) {
    auto head = new HeatGroup();
    auto tail = new HeatGroup();
    head->next = tail;
    tail->prev = head;
    group_queue_.heads_[i] = head;
    group_queue_.tails_[i] = tail;
  }

  LayerHeatBound[0] = 0.0;
  for (size_t i = 1; i < MAX_LAYERS + 1; ++i) {
    LayerHeatBound[i] =
        (std::pow(Coeff, i * LayerTsInterval) - 1) / (Coeff - 1);
  }

  for (size_t i = 0; i < 32; ++i) {
    Timestamps::DecayFactor[i] = 1.0 / std::pow(Coeff, i * LayerTsInterval);
  }
}

void HeatGroupManager::StartHeatThread() {
  heat_processor_ =
      std::thread(&HeatGroupManager::BGWorkProcessHeatGroup, this);
}

void HeatGroupManager::StopHeatThread() {
  AddOperation(nullptr, kOperationStop, true);
  heat_processor_.join();
}

void HeatGroupManager::BGWorkProcessHeatGroup() {
  bool needStop = false;
  while (!needStop) {
    auto operation = group_operations_.pop_front();
    switch(operation.op) {
      case kOperatorSplit:
        SplitGroup(operation.target);
        break;
      case kOperatorMove:
        MoveGroup(operation.target);
        break;
      case kOperatorLevelDown:
        ForceGroupLevelDown();
        break;
      case kOperationChooseCompaction:
        ChooseCompaction();
        break;
      case kOperationStop:
        needStop = true;
        break;
    }
  }
}

void HeatGroupManager::AddOperation(
    HeatGroup* group, GroupOperator op, bool high_pri) {
  high_pri ? group_operations_.emplace_front(group, op) :
           group_operations_.emplace_back(group, op);
}

void HeatGroupManager::InsertIntoLayer(HeatGroup* inserted, int level) {
  if (level >= group_queue_.totalLayers) {
    group_queue_.totalLayers = level + 1;
  }

  auto tail = group_queue_.tails[level];
  inserted->prev = tail->prev;
  inserted->next = tail;
  tail->prev->next = inserted;
  tail->prev = inserted;
  inserted->in_base_layer_ = level == BASE_LAYER;
}

void HeatGroupManager::MoveGroup(HeatGroup* group) {
  auto status = group->status_.load(std::memory_order_relaxed);
  if (unlikely(status == kGroupWaitSplit || status == kGroupCompaction)) {
    return;
  }
  int new_level = ChooseGroupLevel(group);
  RemoveFromQueue(group);
  InsertIntoLayer(group, new_level);
  group->status_.store(kGroupNone, std::memory_order_relaxed);
}

void HeatGroupManager::MoveAllGroupsToLayer(int from, int to) {
  if (group_queue_.heads[from]->next == group_queue_.tails[from]) {
    return;
  }

  auto moved_first = group_queue_.heads[from]->next;
  auto moved_last = group_queue_.tails[from]->prev;
  group_queue_.heads[from]->next = group_queue_.tails[from];
  group_queue_.tails[from]->prev = group_queue_.heads[from];

  auto tail = group_queue_.tails[to];
  auto prev = tail->prev;
  prev->next = moved_first;
  tail->prev = moved_last;
  moved_first->prev = prev;
  moved_last->next = tail;
}

void HeatGroupManager::GroupLevelDown() {
  int layer = 0;
  for (; layer < MAX_LAYERS; ++layer) {
    if (group_queue_.heads[layer]->next != group_queue_.tails[layer]) {
      break;
    }
  }

  if (layer == 0 || layer == MAX_LAYERS) {
    return;
  }

  GlobalDecay.fetch_add(layer * LayerTsInterval, std::memory_order_relaxed);
  //printf("GroupLevelDown: GlobalDecay increased by %d * 100\n", layer);
  group_queue_.totalLayers -= layer;
  for (int l = layer; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - layer);
  }
}

void HeatGroupManager::ForceGroupLevelDown() {
  printf("ForceGroupLevelDown: GlobalDecay increased by 200\n");

  if (group_queue_.totalLayers < 6) {
    return;
  }

  group_queue_.totalLayers -= 2;
  MoveAllGroupsToLayer(1, 0);
  for (int l = 2; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - 2);
  }
}

void HeatGroupManager::TestChooseCompaction() {
  for (int layer = BASE_LAYER; layer < MAX_LAYERS; ++layer) {
    auto group = group_queue_.heads[layer]->next;
    while (group != group_queue_.tails[layer]) {
      auto size = group->group_size_.load();
      if (size > (1 >> 20)) {
        compactor_->Notify(group);
        return;
      }
      group = group->next;
    }
  }
  assert(false);
}

void HeatGroupManager::ChooseCompaction() {
  for (int num_try = 0; num_try < 3; ++num_try) {
    GroupLevelDown();
    auto group = group_queue_.heads[0]->next;
    HeatGroup *nextGroup = group->next;
    if (group == group_queue_.tails[0]) {
      break;
    }

    while (group != group_queue_.tails[0]) {
      RemoveFromQueue(group);
      int newLevel = ChooseGroupLevel(group);
      if (newLevel == 0) {
        InsertIntoLayer(group, TEMP_LAYER);
        group->status_.store(kGroupCompaction, std::memory_order_relaxed);
        compactor_->Notify(group);
        //printf("Found candidate after %d round of search.\n", num_try);
        return;
      } else {
        InsertIntoLayer(group, newLevel);
        auto oldStatus = group->status_.load(std::memory_order_relaxed);
        auto newStatus = oldStatus == kGroupWaitSplit ? kGroupWaitSplit : kGroupNone;
        group->status_.store(newStatus, std::memory_order_relaxed);
      }

      group = nextGroup;
      nextGroup = nextGroup->next;
    }
  }
  compactor_->Notify(nullptr);
}

void HeatGroupManager::SetCompactor(Compactor* compactor) {
  compactor_ = compactor;
}

} // namespace ROCKSDB_NAMESPACE
