//
// Created by joechen on 2022/4/3.
//

#include "heat_group.h"

#include <mutex>
#include <queue>
#include <cmath>
#include <cassert>

#include "macros.h"
#include "timestamp.h"
#include "global_memtable.h"
#include "concurrent_queue.h"
#include "compactor.h"

namespace ROCKSDB_NAMESPACE {

static int32_t GlobalDecr = 0;

static float   LayerHeatBound[MAX_LAYERS + 1];

float          DecayFactor[32];

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

  static float multiplier[48] = {
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
      268435456.0, 536870912.0, 1073741824.0, 2147483648.0
  };

  ts += 512;
  return ts >= 0 ? base[ts & 31] * multiplier[ts >> 5] : 0.0f;
}

int32_t GetGlobalDec() {
  return GlobalDecr;
}

void EstimateLowerAndUpperBound(Timestamps &ts, float &lower_bound, float &upper_bound) {
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

void Timestamps::DecayHeat() {
  auto global_dec = GetGlobalDec();
  if (last_global_dec_ == global_dec) {
    return;
  }

  int delta = global_dec - last_global_dec_;
  last_global_dec_ = GlobalDecr;

#ifdef USE_AVX512F
  _mm256_store_epi32(timestamps,
                     _mm256_sub_epi32(_mm256_set1_epi32(global_dec),
                                      _mm256_loadu_epi32(timestamps)));
#else
  _mm_storeu_si128(
      (__m128i_u *)timestamps,
      _mm_sub_epi32(
          _mm_loadu_si128((__m128i_u *)timestamps),
          _mm_set1_epi32(delta)));
  _mm_storeu_si128(
      (__m128i_u *)(timestamps + 4),
      _mm_sub_epi32(
          _mm_loadu_si128((__m128i_u *)(timestamps + 4)),
          _mm_set1_epi32(delta)));
#endif
  accumulate_ *= DecayFactor[delta / LayerTsInterval];
}

void Timestamps::UpdateHeat() {
  std::lock_guard<SpinLock> lk(update_lock_);
  DecayHeat();
  auto cur_ts = GetTimestamp();
  if (cur_ts <= last_ts_) {
    return;
  }

  last_ts_ = cur_ts;
  cur_ts -= last_global_dec_;
  if (size_ < 8) {
    timestamps[size_++] = cur_ts;
    return;
  }

  accumulate_ += CalculateHeat(timestamps[last_insert_]);
  timestamps[last_insert_++] = cur_ts;
  last_insert_ %= 8;
}

/*----------------------------------------------------------------------*/

HeatGroup *NewHeatGroup() {
  return new HeatGroup();
}

void GroupInsertNewNode(InnerNode *node, InnerNode* inserted) {
  std::vector<InnerNode*> inserts{inserted};
  GroupInsertNewNode(node, inserts);
}

void GroupInsertNewNode(InnerNode *last, std::vector<InnerNode*> &inserts) {
  while (true) {
    auto heatGroup = last->heat_group_;
    if (!heatGroup || !heatGroup->insertNewNode(last, inserts)) {
      _mm_pause();
      continue;
    }
    break;
  }
}

bool HeatGroup::insertNewNode(InnerNode *last, std::vector<InnerNode*> &inserts) {
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

void HeatGroup::updateSize(int32_t size) {
  // This is just a rough estimation, so we don't acquire lock.

  auto curSize = group_size_.fetch_add(size, std::memory_order_relaxed);
  auto curStatus = status_.load(std::memory_order_relaxed);
  if (curStatus != kGroupNone) {
    return;
  }

  GroupStatus newStatus = kGroupNone;
  GroupOperator op;
  if (in_base_layer_ && curSize > GroupMinSize) {
    newStatus = kGroupWaitMove;
    op = kOperatorMove;
  } else if (curSize > GroupSplitThreshold) {
    newStatus = kGroupWaitSplit;
    op = kOperatorSplit;
  }

  if (newStatus != kGroupNone &&
      status_.compare_exchange_strong(curStatus, newStatus, std::memory_order_relaxed)) {
    AddOperation(this, op);
  }
}

/*----------------------------------------------------------------------*/
// GlobalQueue

static MultiLayerGroupQueue GlobalQueue;

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

int ChooseGroupLevel(HeatGroup *group) {
  if (group->group_size_ < GroupMinSize) {
    return -1;
  }

  auto begin = group->first_node_;
  auto end = group->last_node_;
  float totalLowerBound = 0.0f;
  float totalUpperBound = 0.0f;
  float lowerBound = 0.0f;
  float upperBound = 0.0f;

  int count = 0;
  while (begin != end) {
    EstimateLowerAndUpperBound(begin->ts, lowerBound, upperBound);
    totalLowerBound += lowerBound;
    totalUpperBound += upperBound;
    begin = begin->next_node_;
    ++count;
  }

  totalLowerBound /= (float)count;
  totalUpperBound /= (float)count;
  int lowerLevel = ChooseLevelByHeat(totalLowerBound);
  int upperLevel = ChooseLevelByHeat(totalUpperBound);
  return (lowerLevel + upperLevel) >> 1;
}

void InitGroupQueue() {
  // Do some pre calculate
  for (size_t i = 0; i < MAX_LAYERS + 2; ++i) {
    auto head = NewHeatGroup();
    auto tail = NewHeatGroup();
    head->next = tail;
    tail->prev = head;
    GlobalQueue.heads_[i] = head;
    GlobalQueue.tails_[i] = tail;
  }

  LayerHeatBound[0] = 0.0;
  for (size_t i = 1; i < MAX_LAYERS + 1; ++i) {
    LayerHeatBound[i] = (std::pow(Coeff, i * LayerTsInterval) - 1) / (Coeff - 1);
  }

  for (size_t i = 0; i < 32; ++i) {
    DecayFactor[i] = 1.0 / std::pow(Coeff, i * LayerTsInterval);
  }
}

void RemoveFromQueue(HeatGroup *group) {
  group->prev->next = group->next;
  group->next->prev = group->prev;
}

void InsertIntoLayer(HeatGroup *inserted, int level) {
  if (level >= GlobalQueue.totalLayers) {
    GlobalQueue.totalLayers = level + 1;
  }

  auto tail = GlobalQueue.tails[level];
  inserted->prev = tail->prev;
  inserted->next = tail;
  tail->prev->next = inserted;
  tail->prev = inserted;
  inserted->in_base_layer_ = level == BASE_LAYER;
}

void MoveGroup(HeatGroup *group) {
  auto status = group->status_.load(std::memory_order_relaxed);
  if (status == kGroupWaitSplit || status == kGroupCompaction) {
    return;
  }
  int newLevel = ChooseGroupLevel(group);
  RemoveFromQueue(group);
  InsertIntoLayer(group, newLevel);
}

/*----------------------------------------------------------------------*/
// Operation Queue

TQueueConcurrent<GroupOperation> GroupOperations;

// Operation with high priority will be put in front, so they will be executed first.
void AddOperation(HeatGroup *group, GroupOperator op, bool highPri) {
  highPri ? GroupOperations.emplace_front(group, op) :
          GroupOperations.emplace_back(group, op);
}

// TODO. Optimize
void SplitGroup(HeatGroup *group) {
  // This group maybe has been chosen to do compaction, so just pass
  if (group->status_.load(std::memory_order_relaxed) != kGroupWaitSplit) {
    return;
  }

  group->lock.lock();

  auto leftStart = group->first_node_;
  auto rightEnd = group->last_node_;
  auto leftEnd = leftStart;

  auto totalSize = group->group_size_.load(std::memory_order_relaxed);
  auto splitPoint = totalSize >> 1;
  int32_t leftSize = 0;

  while (leftEnd != rightEnd) {
    leftSize += leftEnd->estimated_size_;
    leftEnd->heat_group_ = group;
    if (leftSize > splitPoint) {
      break;
    }
    leftEnd = leftEnd->next_node_;
  }

  // TODO: fix this corner case
  assert(leftEnd != rightEnd);

  group->last_node_ = leftEnd;
  group->group_size_ = totalSize - leftSize;

  auto *rightGroup = NewHeatGroup();
  rightGroup->lock.lock();
  auto rightStart = leftEnd->next_node_;
  rightGroup->first_node_ = rightStart;
  rightGroup->last_node_ = rightEnd;
  rightGroup->group_size_ = totalSize - leftSize;
  while (rightStart != rightEnd) {
    rightStart->heat_group_ = rightGroup;
    rightStart = rightStart->next_node_;
  }

  MoveGroup(group);
  InsertIntoLayer(rightGroup, ChooseGroupLevel(rightGroup));

  rightGroup->status_.store(kGroupNone, std::memory_order_relaxed);
  group->status_.store(kGroupNone, std::memory_order_relaxed);
  rightGroup->lock.unlock();
  group->lock.unlock();
}

void MoveAllGroupsToLayer(int from, int to) {
  if (GlobalQueue.heads[from]->next == GlobalQueue.tails[from]) {
    return;
  }

  auto movedFirst = GlobalQueue.heads[from]->next;
  auto movedLast = GlobalQueue.tails[from]->prev;
  GlobalQueue.heads[from]->next = GlobalQueue.tails[from];
  GlobalQueue.tails[from]->prev = GlobalQueue.heads[from];

  auto tail = GlobalQueue.tails[to];
  auto prev = tail->prev;
  prev->next = movedFirst;
  tail->prev = movedLast;
  movedFirst->prev = prev;
  movedLast->next = tail;
}

void GroupLevelDown() {
  int layer = 0;
  for (; layer < MAX_LAYERS; ++layer) {
    if (GlobalQueue.heads[layer]->next != GlobalQueue.tails[layer]) {
      break;
    }
  }

  if (layer == 0 || layer == MAX_LAYERS) {
    return;
  }

  GlobalDecr += (layer * LayerTsInterval);
  printf("GroupLevelDown: GlobalDecr increased by %d * 100\n", layer);
  GlobalQueue.totalLayers -= layer;
  for (int l = layer; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - layer);
  }
}

// Different from GroupLevelDown,
// groups in layer1 will be moved to layer0 even if layer0 is not empty
void ForceGroupLevelDown() {
  if (GlobalQueue.totalLayers < 6) {
    return;
  }

  GlobalDecr += LayerTsInterval;
  GlobalQueue.totalLayers -= 1;
  printf("ForceGroupLevelDown: GlobalDecr increased by 1 * 100\n");

  for (int l = 1; l < MAX_LAYERS; ++l) {
    MoveAllGroupsToLayer(l, l - 1);
  }
}

void ChooseCompaction() {
  GroupLevelDown();
  auto group = GlobalQueue.heads[0]->next;
  HeatGroup *nextGroup = group->next;
  while (group != GlobalQueue.tails[0]) {
    RemoveFromQueue(group);
    int newLevel = ChooseGroupLevel(group);
    if (newLevel == 0) {
      InsertIntoLayer(group, TEMP_LAYER);
      group->status_.store(kGroupCompaction, std::memory_order_relaxed);
      GetCompactor().Notify(group);
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

  GetCompactor().Notify(nullptr);
}

/*----------------------------------------------------------------------*/
// HeatGroup thread

bool LoopStop = false;

void HeatGroupLoop() {
  bool needStop = false;
  while (!needStop) {
    auto operation = GroupOperations.pop_front();

    auto group = operation.target;

    switch(operation.op) {
      case kOperatorSplit:
        SplitGroup(group);
        break;
      case kOperatorMove:
        MoveGroup(group);
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
  LoopStop = true;
}

void WaitHeatGroupLoopStop() {
  AddOperation(nullptr, kOperationStop);
  while (!LoopStop) {
    _mm_pause();
  }
}

} // namespace ROCKSDB_NAMESPACE
