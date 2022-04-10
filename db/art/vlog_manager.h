//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include <db/dbformat.h>
#include "concurrent_queue.h"

namespace ROCKSDB_NAMESPACE {

#define SEG_HDR_SIZE 128

// Size of vlog file
const constexpr int64_t VLogFileSize = 4ULL << 30; // 4G

// Size of vlog segment
const constexpr int64_t VLogSegmentSize = 1 << 20; // 1M

const constexpr int64_t SegmentIDMask = 0x7ffffffffff00000;

const constexpr int64_t SegmentOffsetMask = 0x00000000000fffff;

// Number of vlog segment
const constexpr int64_t VLogSegmentNum = VLogFileSize / VLogSegmentSize;

const constexpr int64_t VLogHeaderSize = VLogSegmentSize / 128;

const constexpr int64_t VLogBitmapSize = VLogHeaderSize - SEG_HDR_SIZE;

// GC is needed when used space of vlog is larger than threshold.
const float ForceGCThreshold = 0.7;

const float CompactedRatioThreshold = 0.7;

struct alignas(CACHE_LINE_SIZE) AlignedLock {
  SpinMutex mutex_;
};

struct VLogSegmentHeader {
  struct alignas(CACHE_LINE_SIZE) {
    uint32_t offset_ = 0;
    uint16_t total_count_ = 0;
    uint16_t compacted_count_ = 0;
  };
  AlignedLock lock;
  uint8_t bitmap_[VLogBitmapSize];
};

class VLogManager {
  friend class Compactor;

 public:
  explicit VLogManager(bool need_recovery = false);

  uint64_t AddRecord(const Slice& slice, uint32_t record_count = 1);

  RecordIndex GetFirstIndex(size_t wal_size);

  void GetKey(uint64_t offset, std::string &key);

  void GetKey(uint64_t offset, Slice &key);

  ValueType GetKeyValue(uint64_t offset, std::string &key, std::string &value,
                        SequenceNumber &seq_num, RecordIndex &index);

  ValueType GetKeyValue(uint64_t offset, std::string &key,
                        std::string &value, SequenceNumber &seq_num);

  ValueType GetKeyValue(uint64_t offset, std::string &key, std::string &value);

 private:
  void RecoverOnRestart();

  void PopSegment();

  void BGWorkGarbageCollection();

  // address of mapped file
  char* pmemptr_;

  char* cur_segment_;

  VLogSegmentHeader* header_;

  uint64_t offset_;

  uint32_t segment_remain_;

  uint16_t total_count_;

  std::atomic<int> num_free_pages_;

  TQueueConcurrent<char *> free_pages_;

  TQueueConcurrent<char *> used_pages_;

  std::thread gc_thread_;

  bool gc_ = false;
};

VLogManager &GetManager();

}  // namespace ROCKSDB_NAMESPACE
