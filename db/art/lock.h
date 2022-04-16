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

struct OptLock {
  std::atomic<uint32_t> type_version_lock_{0b100};

  uint32_t ReadLockOrRestart(bool &need_restart);

  void WriteLockOrRestart(bool &need_restart);

  void UpgradeToWriteLockOrRestart(uint32_t &version, bool &need_restart);

  void UpgradeToWriteLock();

  void WriteUnlock(bool reverse = false);

  void CheckOrRestart(uint32_t start_read, bool &need_restart) const;

  void ReadUnlockOrRestart(uint32_t start_read, bool &need_restart) const;

  void WriteUnlockObsolete();

  uint32_t AwaitNodeUnlocked();

  static bool IsLocked(uint32_t version);

  static bool IsObsolete(uint32_t version);
};

}  // namespace ROCKSDB_NAMESPACE