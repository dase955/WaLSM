//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <rocksdb/rocksdb_namespace.h>
#include <util/mutexlock.h>
#include "lock.h"

namespace ROCKSDB_NAMESPACE {

#define TS_RESERVE_NUMBER 8

static_assert(TS_RESERVE_NUMBER % 8 == 0,
              "reserved number of ts must be a multiple of 8");

struct Timestamps {
  int32_t timestamps[TS_RESERVE_NUMBER] = {0};
  int32_t last_ts_;
  int32_t last_global_dec_;
  uint8_t size_;
  uint8_t last_insert_;
  float   accumulate_;

  // Hold lock when trying to update heat or update global timestamp
  SpinMutex update_lock_;

  uint8_t padding[12];

  Timestamps();

  Timestamps(const Timestamps& rhs);

  Timestamps& operator=(const Timestamps& rhs);

  Timestamps Copy();

  void Merge(Timestamps& rhs);

  void DecayHeat();

  // Return true if timestamp is updated.
  bool UpdateHeat();

  float GetTotalHeat();

  void EstimateBound(float& lower_bound, float& upper_bound);

  static int factor;
};

static_assert(sizeof(Timestamps) == 64, "sizeof(Timestamps) != 64!");

int32_t GetTimestamp();

void ResetTimestamp();

} // namespace ROCKSDB_NAMESPACE