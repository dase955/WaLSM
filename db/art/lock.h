// Optimistic lock coupling is used for concurrency control of ART
// Spinlock is used to protect some statuses in ART.
// Code copied from
// https://github.com/wangziqi2016/index-microbench/blob/master/BTreeOLC/BTreeOLC_child_layout.h
// and
// https://rigtorp.se/spinlock/

#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <pthread.h>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

// TODO: use first bit to indicate whether we can read from node
// eg. when node is being split, we can still read data from node
// even if OptLock is locked. When node is being compacted, we cannot
// read data from node because data may have been corrupted.
struct OptLock {
  std::atomic<uint32_t> type_version_lock_{0b100};

  static bool IsLocked(uint32_t version) {
    return ((version & 0b10) == 0b10);
  }

  bool TryLock(uint32_t& version);

  void lock();

  void unlock(bool reverse = false);

  uint32_t AwaitUnlocked();

  bool CheckOrRestart(uint32_t start_read) const;

  uint32_t GetCurrentVersion() {
    return type_version_lock_.load(std::memory_order_relaxed);
  }
};

// It is just a copy of RWLatch, but function names are changed.
class SharedMutex {
 public:
  SharedMutex();
  // No copying allowed
  SharedMutex(const SharedMutex&) = delete;
  void operator=(const SharedMutex&) = delete;

  ~SharedMutex();

  void lock_shared();
  void unlock_shared();
  void lock();
  void unlock();

 private:
  pthread_rwlock_t mu_; // the underlying platform mutex
};

template<typename T>
class shared_lock {
 public:
  shared_lock(T& mutex)
      : mutex_(mutex) {
    mutex_.lock_shared();
  };

  ~shared_lock() {
    mutex_.unlock_shared();
  }

 private:
  T& mutex_;
};

class PriorityLock {
 public:
  void LockHighPri() {
    std::unique_lock<std::mutex> lck(mtx_);
    high_pri_locked_.store(true, std::memory_order_release);
    while (low_pri_count_.load(std::memory_order_acquire) > 0) {
      high_pri_cv_.wait(lck);
    }
  }

  void UnlockHighPri() {
    std::unique_lock<std::mutex> lck(mtx_);
    high_pri_locked_.store(false, std::memory_order_release);
    low_pri_cv_.notify_all();
  }

  void LockLowPri() {
    std::unique_lock<std::mutex> lck(mtx_);
    while (high_pri_locked_.load(std::memory_order_acquire)) {
      low_pri_cv_.wait(lck);
    }
    low_pri_count_.fetch_add(1, std::memory_order_release);
  }

  void UnlockLowPri() {
    std::unique_lock<std::mutex> lck(mtx_);
    auto count = low_pri_count_.fetch_sub(1, std::memory_order_release);
    if (count == 1 && high_pri_locked_.load(std::memory_order_acquire)) {
      high_pri_cv_.notify_one();
    }
  }

 private:
  std::mutex mtx_;

  std::condition_variable high_pri_cv_;

  std::condition_variable low_pri_cv_;

  std::atomic_bool high_pri_locked_{false};

  std::atomic_int low_pri_count_{0};
};

}  // namespace ROCKSDB_NAMESPACE