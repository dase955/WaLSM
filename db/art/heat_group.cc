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
#include "nvm_node.h"
#include "timestamp.h"
#include "global_memtable.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

std::atomic<int32_t> GlobalDecay{0};

float LayerHeatBound[MAX_LAYERS + 1];

/*----------------------------------------------------------------------*/

int HeatGroup::group_min_size_;

int HeatGroup::group_split_threshold_;

int HeatGroup::force_decay_waterline_;

int HeatGroup::layer_ts_interval_;

float HeatGroup::decay_factor_[32] = {0};

HeatGroup::HeatGroup(InnerNode* initial_node)
    : group_size_(0),
      first_node_(initial_node), last_node_(initial_node),
      status_(kGroupNone), in_base_layer_(false),
      next(nullptr), prev(nullptr) {}

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

  for (auto& inserted : inserts) {
    if (!inserted->heat_group_) {
      inserted->heat_group_ = this;
    }
  }

  return true;
}

void HeatGroup::UpdateSize(int32_t size) {
  // This is just a rough estimation, so we don't acquire lock.

  GroupStatus cur_status;
  auto cur_size = group_size_.fetch_add(size, std::memory_order_relaxed);
  UpdateTotalSize(size);

  if (cur_size < group_min_size_ ||
      (cur_status = status_.load(std::memory_order_relaxed)) != kGroupNone) {
    return;
  }

  GroupStatus new_status = kGroupNone;
  GroupOperator op;

  if (in_base_layer_ && cur_size > group_min_size_) {
    new_status = kGroupWaitMove;
    op = kOperatorMove;
  } else if (cur_size > group_split_threshold_) {
    new_status = kGroupWaitSplit;
    op = kOperatorSplit;
  }

  if (new_status != kGroupNone &&
      status_.compare_exchange_strong(
          cur_status, new_status, std::memory_order_relaxed)) {
    group_manager_->AddOperation(this, op);
  }
}

void HeatGroup::ResetSize() {
  auto cur_size = group_size_.load(std::memory_order_acquire);
  group_size_.fetch_add(-cur_size, std::memory_order_release);
  UpdateTotalSize(-cur_size);
}

void HeatGroup::UpdateHeat() {
  if (ts.UpdateHeat()) {
    MaybeScheduleHeatDecay(ts.last_ts_);
  }
}

inline void HeatGroup::MaybeScheduleHeatDecay(int32_t last_ts) {
  int32_t decay = GlobalDecay.load(std::memory_order_relaxed);
  if (last_ts - decay > force_decay_waterline_ &&
      GlobalDecay.compare_exchange_strong(
          decay, decay + layer_ts_interval_ * 2, std::memory_order_relaxed)) {
    group_manager_->AddOperation(nullptr, kOperatorLevelDown, true);
  }
}

/*----------------------------------------------------------------------*/

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
  if (group->group_size_.load(std::memory_order_relaxed)
      < HeatGroup::group_min_size_) {
    return -1;
  }

  float lower_bound = 0.0f;
  float upper_bound = 0.0f;

  group->ts.EstimateBound(lower_bound, upper_bound);
  int lower_level = ChooseLevelByHeat(lower_bound);
  int upper_level = ChooseLevelByHeat(upper_bound);
  return (lower_level + upper_level) >> 1;
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

HeatGroupManager::HeatGroupManager(const DBOptions& options) {
  HeatGroup::group_min_size_ = options.group_min_size;
  HeatGroup::group_split_threshold_ = options.group_split_threshold;
  HeatGroup::layer_ts_interval_ = options.layer_ts_interval;
  HeatGroup::force_decay_waterline_ = options.timestamp_waterline;

  InitGroupQueue(options.heat_update_coeff);
  heat_processor_ =
      std::thread(&HeatGroupManager::BGWorkProcessHeatGroup, this);
}

HeatGroupManager::~HeatGroupManager() {
  AddOperation(nullptr, kOperationStop, true);
  heat_processor_.join();
}

void HeatGroupManager::InitGroupQueue(float coeff) {
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
        (float)(std::pow(coeff, i * HeatGroup::layer_ts_interval_) - 1)
        / (coeff - 1.0f);
  }

  for (size_t i = 0; i < 32; ++i) {
    HeatGroup::decay_factor_[i] =
        1.0f / (float)std::pow(coeff, i * HeatGroup::layer_ts_interval_);
  }
}

/*void HeatGroupManager::StartHeatThread() {

}

void HeatGroupManager::StopHeatThread() {
  AddOperation(nullptr, kOperationStop, true);
  heat_processor_.join();
}*/

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
        ChooseCompaction(4);
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

void HeatGroupManager::SplitGroup(HeatGroup* group) {
  // This group maybe has been chosen to do compaction, so just pass
  if (group->status_.load(std::memory_order_relaxed) != kGroupWaitSplit) {
    return;
  }

  std::lock_guard<std::mutex> lk(group->lock);

  auto cur_ts = group->ts.Copy();

  auto left_start = group->first_node_;
  auto right_end = group->last_node_;
  auto left_end = left_start;

  auto stored_total_size = group->group_size_.load(std::memory_order_acquire);
  auto split_point = stored_total_size / 2;
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

  // For each heat group, we need a dummy start node
  // which doesn't store any data. So we can switch between
  // nvm node and its backup node
  InnerNode* dummy_right_start = AllocateLeafNode(0, 0, nullptr);
  SET_GROUP_START(dummy_right_start->status_);
  SET_NON_LEAF(dummy_right_start->status_);
  SET_TAG(dummy_right_start->nvm_node_->meta.header, GROUP_START_TAG);
  InsertInnerNode(left_end, dummy_right_start);

  right_group->ts = cur_ts;
  auto right_start = dummy_right_start;
  right_group->first_node_ = right_start;
  right_group->last_node_ = right_end;
  right_group->group_manager_ = group->group_manager_;
  int32_t right_size = 0;
  while (right_start != right_end) {
    right_start->heat_group_ = right_group;
    right_size += right_start->estimated_size_;
    right_start = right_start->next_node_;
  }
  right_end->heat_group_ = right_group;
  right_size += right_end->estimated_size_;
  right_group->group_size_ = right_size;

  RemoveFromQueue(group);
  group->group_manager_->InsertIntoLayer(group, ChooseGroupLevel(group));
  group->group_manager_->InsertIntoLayer(right_group, ChooseGroupLevel(right_group));

  right_group->status_.store(kGroupNone, std::memory_order_release);
  group->status_.store(kGroupNone, std::memory_order_release);
  right_group->lock.unlock();

  printf("Split group %p %p %d %d %d\n", group, right_group,
         left_size, right_size, stored_total_size);
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

  GlobalDecay.fetch_add(layer * HeatGroup::layer_ts_interval_, std::memory_order_relaxed);
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
  for (int layer = 0; layer < MAX_LAYERS; ++layer) {
    auto group = group_queue_.heads[layer]->next;
    while (group != group_queue_.tails[layer]) {
      auto size = group->group_size_.load();
      if (size > (1 >> 20)) {
        RemoveFromQueue(group);
        InsertIntoLayer(group, TEMP_LAYER);
        //compactor_->Notify(groups);
        return;
      }
      group = group->next;
    }
  }
  assert(false);
}

void HeatGroupManager::ChooseCompaction(size_t num_chosen) {
  std::vector<HeatGroup*> groups;
  for (int num_try = 0; num_try < 2; ++num_try) {
    GroupLevelDown();
    auto group = group_queue_.heads[0]->next;
    HeatGroup * next_group = group->next;
    if (group == group_queue_.tails[0]) {
      break;
    }

    while (group != group_queue_.tails[0]) {
      RemoveFromQueue(group);
      int new_level = ChooseGroupLevel(group);
      if (new_level == 0) {
        InsertIntoLayer(group, TEMP_LAYER);
        group->status_.store(kGroupCompaction, std::memory_order_relaxed);
        groups.push_back(group);
      } else {
        InsertIntoLayer(group, new_level);
        auto oldStatus = group->status_.load(std::memory_order_relaxed);
        auto newStatus = oldStatus == kGroupWaitSplit ? kGroupWaitSplit : kGroupNone;
        group->status_.store(newStatus, std::memory_order_relaxed);
      }

      if (groups.size() == num_chosen) {
        compactor_->Notify(groups);
        return;
      }

      group = next_group;
      next_group = next_group->next;
    }
  }
  compactor_->Notify(groups);
}

void HeatGroupManager::SetCompactor(Compactor* compactor) {
  compactor_ = compactor;
}

} // namespace ROCKSDB_NAMESPACE
