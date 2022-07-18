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
#include "node_allocator.h"
#include "logger.h"

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
      first_node_(initial_node), last_node_(initial_node), status_(kGroupNone),
      in_base_layer(false), in_temp_layer(false), is_removed(false),
      next_seq(nullptr), prev_seq(nullptr),
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

  if ((in_base_layer || in_temp_layer)
      && cur_size > group_min_size_) {
    new_status = kGroupWaitMove;
    op = kOperatorMove;
  } else if (cur_size > group_split_threshold_) {
    new_status = kGroupWaitSplit;
    op = kOperatorSplit;
  }

  if (new_status != kGroupNone &&
      status_.compare_exchange_strong(
          cur_status, new_status, std::memory_order_relaxed)) {
    group_manager_->AddOperation(this, op, op == kOperatorMove);
  }
}

void HeatGroup::UpdateSqueezedSize(int32_t squeezed_size) {
  group_size_.fetch_sub(squeezed_size, std::memory_order_relaxed);
  UpdateTotalSize(-squeezed_size);
  UpdateTotalSqueezedSize(squeezed_size);
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

int ChooseGroupLevelByHeat(HeatGroup* group) {
#ifdef BOUND_ESTIMATION
  float lower_bound = 0.0f;
  float upper_bound = 0.0f;

  group->ts.EstimateBound(lower_bound, upper_bound);
  int lower_level = ChooseLevelByHeat(lower_bound);
  int upper_level = ChooseLevelByHeat(upper_bound);
  return (lower_level + upper_level) >> 1;
#else
  return ChooseLevelByHeat(group->ts.GetTotalHeat());
#endif
}

int ChooseGroupLevel(HeatGroup* group) {
  if (group->group_size_.load(std::memory_order_relaxed)
      < HeatGroup::group_min_size_) {
    return BASE_LAYER;
  }

  return ChooseGroupLevelByHeat(group);
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

  for (size_t i = 0; i < MAX_LAYERS + 2; ++i) {
    auto head = new HeatGroup();
    auto tail = new HeatGroup();
    head->next = tail;
    tail->prev = head;
    group_queue_.heads_[i] = head;
    group_queue_.tails_[i] = tail;
  }

  // Do some pre calculation
  auto coeff = options.heat_update_coeff;

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

void HeatGroupManager::Reset() {
  StopThread();
  group_operations_.clear();
  for (size_t level = 0; level < MAX_LAYERS + 2; ++level) {
    auto group = group_queue_.heads_[level]->next;
    while (group != group_queue_.tails_[level]) {
      auto next_group = group->next;
      delete group->first_node_;
      delete group;
      group = next_group;
    }
    group_queue_.heads_[level]->next = group_queue_.tails_[level];
    group_queue_.tails_[level]->prev = group_queue_.heads_[level];
  }
  group_queue_.total_layers = 1;
  StartThread();
}

void HeatGroupManager::BGWork() {
  Compactor* compactor = nullptr;
  while (!thread_stop_) {
    if (group_operations_.size() == 0) {
      // TryMergeBaseLayerGroups();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    auto operation = group_operations_.pop_front();
    if (unlikely(operation.target && operation.target->is_removed)) {
      continue;
    }

    switch(operation.op) {
      case kOperatorSplit:
        SplitGroup(operation.target);
        break;
      case kOperatorMove:
      case kOperatorMerge:
        MoveGroup(operation.target);
        break;
      case kOperatorLevelDown:
        ForceGroupLevelDown();
        break;
      case kOperationChooseCompaction:
        compactor = (Compactor*)operation.arg;
        ChooseCompaction(compactor, compactor->GetNumParallelCompaction());
        break;
      case kOperationFlushAll:
        ChooseFirstGroup((Compactor*)operation.arg);
        break;
    }
  }
}

void HeatGroupManager::AddOperation(
    HeatGroup* group, GroupOperator op, bool high_pri, void* arg) {
  high_pri ? group_operations_.emplace_front(group, op, arg) :
           group_operations_.emplace_back(group, op, arg);
}

void HeatGroupManager::InsertIntoLayer(HeatGroup* inserted, int level) {
  if (level >= group_queue_.total_layers) {
    group_queue_.total_layers = level + 1;
  }

  auto tail = group_queue_.tails[level];
  inserted->prev = tail->prev;
  inserted->next = tail;
  tail->prev->next = inserted;
  tail->prev = inserted;
  inserted->in_base_layer = level == BASE_LAYER;
  inserted->in_temp_layer = level == TEMP_LAYER;
}

bool HeatGroupManager::MergeNextGroup(HeatGroup* group, HeatGroup* next_group) {
  if (unlikely(!group ||
      group->status_.load(std::memory_order_relaxed) == kGroupCompaction)) {
    return false;
  }

  // lock this group and its last node
  std::lock_guard<std::mutex> group_lk(group->lock);
  InnerNode* last_node = group->last_node_;
  std::lock_guard<RWSpinLock> link_lk(last_node->link_lock_);
  InnerNode* next_start_node = last_node->next_node;

  if (unlikely(NOT_GROUP_START(next_start_node))) {
    assert(!next_start_node->heat_group_ ||
           next_start_node->heat_group_ == group);
    return false;
  }

  // Group has been split, just return
  if (unlikely(group->next_seq != next_group ||
      next_group->status_.load(std::memory_order_relaxed) == kGroupCompaction)) {
    return false;
  }

  std::lock_guard<std::mutex> next_group_lk(next_group->lock);
  assert(next_group != group && next_group->first_node_ == next_start_node);
  assert(IS_GROUP_START(next_start_node));
  assert(next_group->last_node_->heat_group_ == next_group);

  assert(group->next_seq == next_group);

  if (!next_group->in_base_layer || !group->in_base_layer ||
      next_group->status_.load(std::memory_order_relaxed) == kGroupCompaction ||
      ChooseGroupLevelByHeat(group) != ChooseGroupLevelByHeat(next_group)) {
    return false;
  }

  int total_size = group->group_size_.load(std::memory_order_relaxed) +
                   next_group->group_size_.load(std::memory_order_relaxed);
  if (total_size > HeatGroup::group_min_size_ * 3 / 2) {
    return false;
  }

  // Merge two group and remove dummy node
  InnerNode* node_after_start = next_start_node->next_node;
  assert(NOT_GROUP_START(node_after_start));
  assert(node_after_start->heat_group_ == next_group);

  last_node->next_node = node_after_start;
  auto next_nvm_node = GetNodeAllocator()->relative(node_after_start->nvm_node_);
  if (GET_TAG(last_node->nvm_node_->meta.header, ALT_FIRST_TAG)) {
    last_node->nvm_node_->meta.next1 = next_nvm_node;
  } else {
    last_node->nvm_node_->meta.next2 = next_nvm_node;
  }
  PERSIST(last_node->nvm_node_, CACHE_LINE_SIZE);

  InnerNode* cur = node_after_start;
  next_group->last_node_->heat_group_ = group;
  while (cur != next_group->last_node_) {
    cur->heat_group_ = group;
    cur = cur->next_node;
  }

  group->group_size_.fetch_add(
      next_group->group_size_.load(std::memory_order_relaxed));
  group->ts.Merge(next_group->ts);
  group->last_node_ = next_group->last_node_;
  group->next_seq = next_group->next_seq;
  group->next_seq->prev_seq = group;

  RemoveFromQueue(group);
  InsertIntoLayer(group, ChooseGroupLevel(group));
  group->status_.store(kGroupNone, std::memory_order_relaxed);

  RemoveFromQueue(next_group);
  next_group->is_removed = true;

  // Free dummy node
  GetNodeAllocator()->DeallocateNode(next_start_node->nvm_node_);
  delete next_start_node;
  return true;
}

void HeatGroupManager::TryMergeBaseLayerGroups() {
  static int idx = 0;
  idx += 2;

  auto group = group_queue_.heads[BASE_LAYER]->next;
  auto end_group = group_queue_.tails[BASE_LAYER];

  int count = 0;
  auto cur = group;
  while (cur != end_group) {
    cur = cur->next;
    ++count;
  }

  count = idx % count;

  while (group != end_group) {
    if (count-- == 0) {
      MergeNextGroup(group, group->next_seq);
      return;
    }
    group = group->next;
  }
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

  if (unlikely(new_level == BASE_LAYER)) {
    if (!MergeNextGroup(group, group->next_seq)) {
      MergeNextGroup(group->prev_seq, group);
    }
  }
}

void HeatGroupManager::SplitGroup(HeatGroup* group) {
  // This group maybe has been chosen to do compaction, so just pass
  if (group->status_.load(std::memory_order_relaxed) != kGroupWaitSplit) {
    return;
  }

  auto right_group = new HeatGroup();

  std::lock_guard<std::mutex> lk(group->lock);
  std::lock_guard<std::mutex> right_lk(right_group->lock);

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
    assert(left_end->next_node);
    left_end = left_end->next_node;
  }

  // TODO: fix this corner case
  assert(left_end != right_end);

  group->last_node_ = left_end;
  group->group_size_ = left_size;

  right_group->next_seq = group->next_seq;
  right_group->prev_seq = group;
  group->next_seq->prev_seq = right_group;
  group->next_seq = right_group;

  // For each heat group, we need a dummy start node
  // which doesn't store any data, it is helpful when switching between
  // nvm_node and backup_nvm_node in compaction.
  InnerNode* dummy_right_start = AllocateLeafNode(0, 0, nullptr);
  SET_GROUP_START(dummy_right_start);
  SET_NON_LEAF(dummy_right_start);
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
    right_start = right_start->next_node;
  }
  right_end->heat_group_ = right_group;
  right_size += right_end->estimated_size_;
  right_group->group_size_ = right_size;

  RemoveFromQueue(group);
  group->group_manager_->InsertIntoLayer(group, ChooseGroupLevel(group));
  group->group_manager_->InsertIntoLayer(right_group, ChooseGroupLevel(right_group));

  right_group->status_.store(kGroupNone, std::memory_order_release);
  group->status_.store(kGroupNone, std::memory_order_release);
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

  if (layer <= 2 || layer == MAX_LAYERS) {
    return;
  }
  layer -= 2;

  GlobalDecay.fetch_add(layer * HeatGroup::layer_ts_interval_, std::memory_order_relaxed);
  group_queue_.total_layers -= layer;
  for (int l = layer; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - layer);
  }
}

void HeatGroupManager::ForceGroupLevelDown() {
  if (group_queue_.total_layers < 6) {
    return;
  }

  group_queue_.total_layers -= 2;
  MoveAllGroupsToLayer(1, 0);
  for (int l = 2; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - 2);
  }
}

bool HeatGroupManager::CheckForCompaction(HeatGroup* group, int cur_level) {
  RemoveFromQueue(group);
  int new_level = ChooseGroupLevel(group);
  if (new_level >= 0 && new_level <= cur_level) {
    InsertIntoLayer(group, TEMP_LAYER);
    group->status_.store(kGroupCompaction, std::memory_order_relaxed);
    return true;
  } else {
    InsertIntoLayer(group, new_level);
    auto oldStatus = group->status_.load(std::memory_order_relaxed);
    auto newStatus = oldStatus == kGroupWaitSplit ? kGroupWaitSplit : kGroupNone;
    group->status_.store(newStatus, std::memory_order_relaxed);
    return false;
  }
}

void HeatGroupManager::ChooseCompaction(
    Compactor* compactor, size_t num_chosen) {
  std::vector<HeatGroup*> chosen_groups;

  // group_queue_.CountGroups();

  for (int num_tries = 0; num_tries < 2; ++num_tries) {
    for (int l = 0; l < 3; ++l) {
      auto group = group_queue_.heads[l]->next;
      auto end_group = group_queue_.tails[l];
      while (group != end_group) {
        HeatGroup* next_group = group->next;
        if (CheckForCompaction(group, l)) {
          chosen_groups.push_back(group);
          if (chosen_groups.size() == num_chosen) {
            compactor->Notify(chosen_groups);
            return;
          }
        }
        group = next_group;
      }
    }
  }

  if (chosen_groups.empty()) {
    GroupLevelDown();
  }

  compactor->Notify(chosen_groups);
}

void HeatGroupManager::ChooseFirstGroup(Compactor* compactor) {
  HeatGroup* group = nullptr;
  for (int level = BASE_LAYER; level < 7; ++level) {
    if (group_queue_.heads[level]->next != group_queue_.tails[level]) {
      group = group_queue_.heads[level]->next;
      break;
    }
  }

  assert(group);
  while (group->prev_seq) {
    group = group->prev_seq;
  }

  auto first_group = group;
  while (group) {
    group->is_removed = true;
    group = group->next_seq;
  }

  std::vector<HeatGroup*> groups{first_group};
  compactor->Notify(groups);
}

} // namespace ROCKSDB_NAMESPACE
