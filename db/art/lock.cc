//
// Created by joechen on 2022/4/3.
//

#include "lock.h"

#include <immintrin.h>

#include "port/port.h"

namespace ROCKSDB_NAMESPACE {

bool OptLock::TryLock(uint32_t& version) {
  if (type_version_lock_.compare_exchange_strong(
          version, version + 0b10, std::memory_order_release)) {
    version = version + 0b10;
    return true;
  }
  else {
    _mm_pause();
    return false;
  }
}

void OptLock::lock() {
  uint32_t version;
  for (size_t tries = 0;; ++tries) {
    version = type_version_lock_.load(std::memory_order_acquire);
    if (!IsLocked(version) &&
        type_version_lock_.compare_exchange_strong(
            version, version + 0b10, std::memory_order_release)) {
      return;
    }
    port::AsmVolatilePause();
    if (tries > 100) {
      std::this_thread::yield();
    }
  }
}

void OptLock::unlock(bool reverse) {
  reverse ? type_version_lock_.fetch_sub(0b10, std::memory_order_release)
          : type_version_lock_.fetch_add(0b10, std::memory_order_release);
}

bool OptLock::CheckOrRestart(uint32_t start_read) const {
  return (start_read != type_version_lock_.load(std::memory_order_acquire));
}

uint32_t OptLock::AwaitUnlocked() {
  uint32_t version;
  for (size_t tries = 0;; ++tries) {
    if (((version = type_version_lock_.load(std::memory_order_acquire))
         & 0b10) != 0b10) {
      return version;
    }
    port::AsmVolatilePause();
    if (tries > 100) {
      std::this_thread::yield();
    }
  }
}

SharedMutex::SharedMutex() {
  pthread_rwlock_init(&mu_, nullptr);
}

SharedMutex::~SharedMutex() {
  pthread_rwlock_destroy(&mu_);
}

void SharedMutex::lock_shared() {
  pthread_rwlock_rdlock(&mu_);
}

void SharedMutex::lock() {
  pthread_rwlock_wrlock(&mu_);
}

void SharedMutex::unlock_shared() {
  pthread_rwlock_unlock(&mu_);
}

void SharedMutex::unlock() {
  pthread_rwlock_unlock(&mu_);
}

}  // namespace ROCKSDB_NAMESPACE