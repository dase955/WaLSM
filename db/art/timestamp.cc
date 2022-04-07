//
// Created by joechen on 2022/4/3.
//

#include "timestamp.h"

#include <atomic>
#include <cstring>

#include "utils.h"
#include "heat_group.h"
#include "heat_group_manager.h"

namespace ROCKSDB_NAMESPACE {

float Timestamps::DecayFactor[32] = {0};

int32_t GetTimestamp() {
  static std::atomic<int32_t> Timestamp{16384};
  return Timestamp.fetch_add(1, std::memory_order_relaxed) >> 14;
}

Timestamps::Timestamps()
    : last_ts_(0), last_global_dec_(0), size_(0),
      last_insert_(0), accumulate_(0.f) {}

Timestamps::Timestamps(const Timestamps &rhs)
    : last_ts_(rhs.last_ts_), last_global_dec_(rhs.last_global_dec_),
      size_(rhs.size_), last_insert_(rhs.last_insert_),
      accumulate_(rhs.accumulate_) {
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

Timestamps Timestamps::Copy() {
  std::lock_guard<SpinMutex> lk(update_lock_);
  Timestamps ts(*this);
  return ts;
}

void Timestamps::DecayHeat() {
  auto global_dec = GlobalDecay.load(std::memory_order_relaxed);
  if (last_global_dec_ == global_dec) {
    return;
  }

  int delta = global_dec - last_global_dec_;
  last_global_dec_ = global_dec;

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
  delta /= LayerTsInterval;
  accumulate_ *= delta < 32 ? DecayFactor[delta] : 0;
}

void Timestamps::UpdateHeat() {
  auto cur_ts = GetTimestamp();
  if (likely(cur_ts <= last_ts_)) {
    return;
  }

  std::lock_guard<SpinMutex> lk(update_lock_);
  if (unlikely(cur_ts <= last_ts_)) {
    return;
  }
  last_ts_ = cur_ts;

  DecayHeat();

  cur_ts -= last_global_dec_;
  if (size_ < 8) {
    timestamps[size_++] = cur_ts;
    return;
  }

  accumulate_ += CalculateHeat(timestamps[last_insert_]);
  timestamps[last_insert_++] = cur_ts;
  last_insert_ %= 8;
}

bool Timestamps::GetCurrentHeatAndTs(
    int32_t &begin_ts, int32_t &mid_ts, int32_t &end_ts, float &heat) {
  std::lock_guard<SpinMutex> lk(update_lock_);
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