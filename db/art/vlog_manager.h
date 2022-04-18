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

struct alignas(CACHE_LINE_SIZE) AlignedLock {
  SpinMutex mutex_;
};

enum SegmentStatus : uint8_t {
  kSegmentFree,     // Initial status
  kSegmentWriting,  // Writing records to segment
  kSegmentWritten,  // Segment is full, and wait for gc
  kSegmentGC,       // Segment is doing gc
};

struct VLogSegmentHeader {
  struct alignas(CACHE_LINE_SIZE) {
    SegmentStatus status_ : 8;
    uint32_t offset_: 24;
    uint16_t total_count_ = 0;
    uint16_t compacted_count_ = 0;
  };
  AlignedLock lock;
  uint8_t bitmap_[];
};

struct GCData {
  uint64_t vptr_;
  std::string record_;
  std::string key_;

  GCData() = default;

  GCData(uint32_t key_start, uint32_t key_length,
         uint64_t vptr, std::string& record)
      : vptr_(vptr), record_(record),
        key_(std::string(record_.data() + key_start, key_length)){
  };
};

class GlobalMemtable;

class VLogManager {
  friend class Compactor;

 public:
  explicit VLogManager(const DBOptions& options);

  ~VLogManager();

  void SetMemtable(GlobalMemtable* mem);

  uint64_t AddRecord(const Slice& slice, uint32_t record_count = 1);

  void GetKey(uint64_t vptr, std::string &key);

  void GetKey(uint64_t vptr, Slice &key);

  ValueType GetKeyValue(uint64_t offset, std::string& key, std::string& value,
                        SequenceNumber& seq_num, RecordIndex& index);

  ValueType GetKeyValue(uint64_t offset, std::string& key,
                        std::string& value, SequenceNumber& seq_num);

  ValueType GetKeyValue(uint64_t offset, std::string& key, std::string& value);

  RecordIndex GetFirstIndex(size_t wal_size) const;

  void FreeQueue();

  void TestGC();

 private:
  void BGWorkGarbageCollection();

  void ChangeStatus(char* segment, SegmentStatus status);

  void PopFreeSegment();

  char* WriteToNewSegment(
      char *segment, std::string& record, uint64_t& new_vptr);

  char* ChooseSegmentToGC();

  std::vector<GCData> ReadAndSortData(char* segment);

  char* pmemptr_;

  char* cur_segment_;

  VLogSegmentHeader* header_;

  uint64_t offset_;

  uint32_t segment_remain_;

  uint16_t count_in_segment_;

  GlobalMemtable* mem_;

  TQueueConcurrent<char*> free_pages_;

  TQueueConcurrent<char*> used_pages_;

  // segment should push into a separate queue after gc,
  // because compaction thread may still need these segments.
  TQueueConcurrent<char*> gc_pages_;

  uint64_t vlog_file_size_;

  uint64_t vlog_segment_size_;

  size_t vlog_segment_num_;

  uint64_t vlog_header_size_;

  uint64_t vlog_bitmap_size_;

  size_t force_gc_ratio_;

  const float compacted_ratio_threshold_ = 0.5;

  // Mutex and condvar for gc thread
  port::Mutex gc_mu_;
  port::CondVar gc_cv_;
  bool thread_stop_;
  std::thread gc_thread_;
  bool gc_ = false;
};

}  // namespace ROCKSDB_NAMESPACE
