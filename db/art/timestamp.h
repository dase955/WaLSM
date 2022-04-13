//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <rocksdb/rocksdb_namespace.h>
#include <util/mutexlock.h>
#include "lock.h"

namespace ROCKSDB_NAMESPACE {

struct Timestamps {
  int32_t timestamps[8] = {0};
  int32_t last_ts_;
  int32_t last_global_dec_;
  uint8_t size_;
  uint8_t last_insert_;
  float   accumulate_;

  // Hold lock when trying to update heat or update global timestamp
  SpinMutex update_lock_;

  Timestamps();

  Timestamps(const Timestamps& rhs);

  Timestamps& operator=(const Timestamps& rhs);

  Timestamps Copy();

  void DecayHeat();

  void UpdateHeat();

  bool GetCurrentHeatAndTs(
      int32_t& begin_ts, int32_t& mid_ts, int32_t& end_ts, float& heat);
};

int32_t GetTimestamp();

} // namespace ROCKSDB_NAMESPACE