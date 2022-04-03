//
// Created by joechen on 2022/4/3.
//

#include "timestamp.h"

#include <atomic>
#include <mutex>

#include "macros.h"
#include "heat_group.h"

namespace ROCKSDB_NAMESPACE {

int32_t GetTimestamp() {
  static ::std::atomic<int32_t> Timestamp{1024};
  static ::std::atomic<int32_t> DecayPoint{DecayThreshold};
  int32_t ts = (Timestamp++) >> 10;
  int32_t p = DecayPoint.load(::std::memory_order_relaxed);
  if (ts > p && DecayPoint.compare_exchange_strong(p, p + LayerTsInterval * 2, ::std::memory_order_release)) {
    AddOperation(nullptr, kOperatorLevelDown, true);
  }
  return ts;
}

Timestamps::Timestamps(): last_ts_(0), last_global_dec_(0), size_(0), last_insert_(0), accumulate_(0.f) {}

Timestamps::Timestamps(const Timestamps &rhs)
    : last_ts_(rhs.last_ts_), last_global_dec_(rhs.last_global_dec_),
      size_(rhs.size_), last_insert_(rhs.last_insert_), accumulate_(rhs.accumulate_) {
  memcpy(timestamps, rhs.timestamps, 32);
}

Timestamps &Timestamps::operator=(const Timestamps &rhs) {
  if (this == &rhs) {
    return *this;
  }

  last_ts_ = rhs.last_ts_;
  last_global_dec_ = rhs.last_global_dec_;
  size_ = rhs.size_;
  last_insert_ = rhs.last_insert_;
  accumulate_ = rhs.accumulate_;
  memcpy(timestamps, rhs.timestamps, 32);
  return *this;
}

Timestamps Timestamps::copy() {
  std::lock_guard<SpinLock> lk(update_lock_);
  Timestamps ts(*this);
  return ts;
}

bool Timestamps::GetCurrentHeatAndTs(int32_t &begin_ts, int32_t &mid_ts, int32_t &end_ts, float &heat) {
  std::lock_guard<SpinLock> lk(update_lock_);
  if (size_ == 0) {
    return false;
  }
  DecayHeat();

  int begin = last_insert_;
  int mid = (begin + size_ / 2) % 8;
  int end = (begin + size_ - 1) % 8;

  begin_ts = timestamps[begin];
  mid_ts = timestamps[mid];
  end_ts = timestamps[end];
  heat = accumulate_;
  return true;
}

} // namespace ROCKSDB_NAMESPACE