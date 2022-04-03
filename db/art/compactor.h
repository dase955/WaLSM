//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <mutex>
#include <condition_variable>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

// Forward declaration
struct HeatGroup;

class Compactor {
 public:
  std::atomic<size_t> total_size_;

  std::mutex mutex_;

  std::condition_variable cond_var_;

  bool m_bStop;

  HeatGroup *chosen_group_ = nullptr;

  int32_t doCompaction();

 public:
  Compactor() : total_size_(0), m_bStop(false) {};

  void updateSize(size_t addedSize);

  void MainLoop();

  void StopLoop();

  void Notify(HeatGroup *heatGroup);
};

Compactor &GetCompactor();

} // namespace ROCKSDB_NAMESPACE