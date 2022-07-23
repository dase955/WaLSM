//
// Created by joechen on 2022/4/3.
//

#pragma once
#include <cstdint>
#include <rocksdb/rocksdb_namespace.h>
#include <rocksdb/slice.h>
#include <db/dbformat.h>
#include <util/autovector.h>
#include "concurrent_queue.h"
#include "utils.h"

namespace ROCKSDB_NAMESPACE {

enum SegmentStatus : uint8_t {
  kSegmentFree,     // Initial status
  kSegmentWriting,  // Writing records to segment
  kSegmentWritten,  // Segment is full, and wait for gc
  kSegmentGC,       // Segment is doing gc
};

struct StatusLock {
  SpinMutex     mutex;
  SegmentStatus status;
};

struct VLogSegmentHeader {
  struct alignas(CACHE_LINE_SIZE) {
    // TODO: remove status from nvm to dram
    SegmentStatus status_to_remove : 8;
    uint32_t offset_: 24;
    uint16_t total_count_ = 0;
    uint16_t compacted_count_ = 0;
  };
  char    padding[64];
  uint8_t bitmap_[];
};

struct alignas(CACHE_LINE_SIZE) GCData {
  std::string record;
  uint64_t    actual_vptr;
  Slice       key;

  GCData() = default;

  GCData(uint32_t key_start, uint32_t key_length,
         uint64_t actual_vptr_, std::string& record_)
      : record(std::move(record_)),
        actual_vptr(actual_vptr_),
        key(record.data() + key_start, key_length) {};
};


class GlobalMemtable;

class VLogManager : public BackgroundThread {
  friend class Compactor;

 public:
  explicit VLogManager(const DBOptions& options, bool recovery = false);

  ~VLogManager() override;

  void Initialize();

  void Recover();

  void Reset();

  void SetMemtable(GlobalMemtable* mem);

  RecordIndex GetFirstIndex(size_t wal_size) const;

  uint64_t AddRecord(const Slice& slice, uint32_t record_count);

  void GetKey(uint64_t vptr, Slice& key);

  void GetKeyIndex(uint64_t vptr, std::string& key, RecordIndex& index);

  ValueType GetKeyValue(uint64_t offset, std::string& key, std::string& value);

  ValueType GetKeyValue(uint64_t offset, std::string& key, std::string& value,
                        SequenceNumber& seq_num);

  ValueType GetKeyValue(uint64_t offset,
                        std::string& key, Slice& value,
                        SequenceNumber& seq_num, RecordIndex& index);

  void UpdateBitmap(
      std::unordered_map<uint64_t, std::vector<RecordIndex>>& all_indexes);

  void UpdateBitmap(std::vector<std::vector<RecordIndex>>& all_indexes);

  void FreeQueue();

  float Estimate();

  void MaybeRewrite(KVStruct& kv_info);

 private:
  void BGWork() override;

  char* GetSegmentFromFreeQueue();

  void PushToUsedQueue(char* segment);

  void PopFreeSegment();

  void WriteToNewSegment(std::string& record, uint64_t& new_vptr);

  char* ChooseSegmentToGC();

  void ReadAndSortData(std::vector<char*>& segments);

  VLogSegmentHeader* GetHeader(size_t index) {
    return (VLogSegmentHeader*)(pmemptr_ + vlog_segment_size_ * index);
  }

  size_t GetIndex(const char* segment) {
    return (segment - pmemptr_) / vlog_segment_size_;
  }

  char* pmemptr_;

  char* cur_segment_;

  VLogSegmentHeader* header_;

  uint32_t segment_remain_;

  GlobalMemtable* mem_;

  TQueueConcurrent<char*> free_segments_;

  TQueueConcurrent<char*> used_segments_;

  // segment should push into a separate queue after gc,
  // because compaction thread may still need these segments.
  TQueueConcurrent<char*> gc_pages_;

  std::vector<GCData> gc_data;

  char* segment_for_gc_;

  // Parameters
  const uint64_t vlog_file_size_;
  const uint64_t vlog_segment_size_;
  const size_t   vlog_segment_num_;
  const uint64_t vlog_header_size_;
  const uint64_t vlog_bitmap_size_;
  const size_t   force_gc_ratio_;
  const float    compacted_ratio_threshold_ = 0.5;

  StatusLock*    segment_statuses_;
};

}  // namespace ROCKSDB_NAMESPACE
