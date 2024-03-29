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


int Timestamps::factor;

static std::atomic<int32_t> Timestamp{1};

int32_t GetTimestamp() {
  return Timestamp.fetch_add(1, std::memory_order_relaxed) >> Timestamps::factor;
}

void ResetTimestamp() {
  Timestamp.store(1 << Timestamps::factor);
}

Timestamps::Timestamps()
    : last_ts_(0), last_global_dec_(0), size_(0),
      last_insert_(0), accumulate_(0.f) {}

Timestamps::Timestamps(const Timestamps& rhs)
    : last_ts_(rhs.last_ts_), last_global_dec_(rhs.last_global_dec_),
      size_(rhs.size_), last_insert_(rhs.last_insert_),
      accumulate_(rhs.accumulate_) {
  memcpy(timestamps, rhs.timestamps, sizeof(int) * TS_RESERVE_NUMBER);
}

Timestamps& Timestamps::operator=(const Timestamps& rhs) {
  if (this == &rhs) {
    return *this;
  }

  last_ts_ = rhs.last_ts_;
  last_global_dec_ = rhs.last_global_dec_;
  size_ = rhs.size_;
  last_insert_ = rhs.last_insert_;
  accumulate_ = rhs.accumulate_;
  memcpy(timestamps, rhs.timestamps, sizeof(int) * TS_RESERVE_NUMBER);
  return *this;
}

Timestamps Timestamps::Copy() {
  std::lock_guard<SpinMutex> lk(update_lock_);
  Timestamps ts(*this);
  return ts;
}

void Timestamps::Merge(Timestamps& rhs) {
  float heat = GetTotalHeat();
  float heat_rhs = rhs.GetTotalHeat();

  std::lock_guard<SpinMutex> update_lk(update_lock_);
  accumulate_ = (heat + heat_rhs) / 2;
}

void Timestamps::DecayHeat() {
  auto global_dec = GlobalDecay.load(std::memory_order_relaxed);
  if (likely(last_global_dec_ >= global_dec)) {
    return;
  }

  int32_t delta = global_dec - last_global_dec_;
  last_global_dec_ = global_dec;

#ifdef USE_AVX512F
  for (int i = 0; i < TS_RESERVE_NUMBER; i += 8) {
    _mm256_store_epi32(timestamps + i,
                       _mm256_sub_epi32(_mm256_set1_epi32(global_dec),
                                        _mm256_loadu_epi32((timestamps + i))));
  }
#else
  for (int i = 0; i < TS_RESERVE_NUMBER; i += 4) {
    _mm_storeu_si128(
        (__m128i_u *)(timestamps + i),
        _mm_sub_epi32(
            _mm_loadu_si128((__m128i_u *)(timestamps + i)),
            _mm_set1_epi32(delta)));
  }
#endif
  accumulate_ *= GetDecayFactor(delta);
}

bool Timestamps::UpdateHeat() {
  auto cur_ts = GetTimestamp();
  if (likely(cur_ts <= last_ts_)) {
    return false;
  }

  std::lock_guard<SpinMutex> update_lk(update_lock_);
  if (unlikely(cur_ts <= last_ts_)) {
    return false;
  }
  last_ts_ = cur_ts;

  DecayHeat();

  cur_ts -= last_global_dec_;
  if (size_ < TS_RESERVE_NUMBER) {
    timestamps[size_++] = cur_ts;
    return true;
  }

  accumulate_ += CalculateHeat(timestamps[last_insert_]);
  timestamps[last_insert_++] = cur_ts;
  last_insert_ %= TS_RESERVE_NUMBER;

  return true;
}

float Timestamps::GetTotalHeat() {
  std::lock_guard<SpinMutex> update_lk(update_lock_);
  DecayHeat();
  for (size_t i = 0; i < size_; ++i) {
    accumulate_ += CalculateHeat(timestamps[i]);
  }
  size_ = last_insert_ = 0;
  return accumulate_;
}

void Timestamps::EstimateBound(float& lower_bound, float& upper_bound) {
  int len;
  float heat;
  int32_t begin_ts, mid_ts, end_ts;

  {
    std::lock_guard<SpinMutex> update_lk(update_lock_);
    DecayHeat();
    heat = accumulate_;

    if (size_ == 0) {
      lower_bound = upper_bound = heat;
      return;
    }

    len = (size_ + 1) / 2;
    begin_ts = timestamps[last_insert_];
    mid_ts = timestamps[(last_insert_ + size_ / 2) % TS_RESERVE_NUMBER];
    end_ts = timestamps[(last_insert_ + size_ - 1) % TS_RESERVE_NUMBER];
  }

  float h1 = CalculateHeat(begin_ts);
  float h2 = CalculateHeat(mid_ts);
  float h3 = CalculateHeat(end_ts);
  lower_bound = heat + (h1 + h2) * (float)len;
  upper_bound = heat + (h2 + h3) * (float)len;
}

} // namespace ROCKSDB_NAMESPACE