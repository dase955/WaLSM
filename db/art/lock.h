// Optimistic lock coupling is used for concurrency control of ART
// Spinlock is used to protect some statuses in ART.
// Code copied from
// https://github.com/wangziqi2016/index-microbench/blob/master/BTreeOLC/BTreeOLC_child_layout.h
// and
// https://rigtorp.se/spinlock/

#pragma once
#include <atomic>
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

  static bool IsObsolete(uint32_t version) {
    return (version & 1) == 1;
  }

  void SetObsolete() {
    type_version_lock_ &= 1;
  }

  void RemoveObsolete() {
    type_version_lock_ &= -2;
  }

  void UpgradeToWriteLockOrRestart(uint32_t& version, bool& need_restart);

  void WriteLock();

  void WriteUnlock(bool reverse = false);

  uint32_t AwaitNodeUnlocked();

  void CheckOrRestart(uint32_t start_read, bool& need_restart) const;

  // Function for read
  uint32_t AwaitNodeReadable();

  uint32_t GetCurrentVersion() {
    return type_version_lock_.load(std::memory_order_relaxed);
  }
};

}  // namespace ROCKSDB_NAMESPACE