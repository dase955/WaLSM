//
// Created by joechen on 2022/4/3.
//

#include "lock.h"

#include <immintrin.h>

namespace ROCKSDB_NAMESPACE {

bool OptLock::IsLocked(uint32_t version) {
  return ((version & 0b10) == 0b10);
}

uint32_t OptLock::ReadLockOrRestart(bool &need_restart) {
  uint32_t version;
  version = type_version_lock_.load(std::memory_order_acquire);
  if (IsLocked(version) || IsObsolete(version)) {
    _mm_pause();
    need_restart = true;
  }
  return version;
}

void OptLock::WriteLockOrRestart(bool &need_restart) {
  uint32_t version;
  version = ReadLockOrRestart(need_restart);
  if (need_restart) {
    return;
  }

  UpgradeToWriteLockOrRestart(version, need_restart);
  if (need_restart) {
    return;
  }
}

void OptLock::UpgradeToWriteLockOrRestart(uint32_t &version, bool &need_restart) {
  if (type_version_lock_.compare_exchange_strong(
          version, version + 0b10, std::memory_order_release)) {
    version = version + 0b10;
  }
  else {
    _mm_pause();
    need_restart = true;
  }
}

void OptLock::UpgradeToWriteLock() {
  uint32_t version;
  while (true) {
    version = AwaitNodeUnlocked();
    if (type_version_lock_.compare_exchange_strong(
            version, version + 0b10, std::memory_order_release)) {
      version = version + 0b10;
      return;
    } else {
      _mm_pause();
    }
  }
}

void OptLock::WriteUnlock(bool reverse) {
  reverse ? type_version_lock_.fetch_sub(0b10, std::memory_order_release)
          : type_version_lock_.fetch_add(0b10, std::memory_order_release);
}

bool OptLock::IsObsolete(uint32_t version) {
  return (version & 1) == 1;
}

void OptLock::CheckOrRestart(uint32_t start_read, bool &need_restart) const {
  ReadUnlockOrRestart(start_read, need_restart);
}

void OptLock::ReadUnlockOrRestart(uint32_t start_read, bool &need_restart) const {
  need_restart = (start_read != type_version_lock_.load(std::memory_order_acquire));
}

void OptLock::WriteUnlockObsolete() {
  type_version_lock_.fetch_add(0b11, std::memory_order_release);
}

uint32_t OptLock::AwaitNodeUnlocked() {
  uint32_t version;
  while (((version = type_version_lock_.load(std::memory_order_acquire)) & 0b10) == 0b10) {
    _mm_pause();
  }
  return version;
}

void SpinLock::lock() {
  for (;;) {
    if (!lock_.exchange(true, std::memory_order_acquire)) {
      break;
    }
    while (lock_.load(std::memory_order_release)) {
      __builtin_ia32_pause();
    }
  }
}

void SpinLock::unlock() {
  lock_.store(false, std::memory_order_release);
}
}  // namespace ROCKSDB_NAMESPACE