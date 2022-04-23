//
// Created by joechen on 2022/4/3.
//

#include "vlog_manager.h"

#include <sys/mman.h>
#include <cassert>
#include <string>
#include <cstring>

#include <util/hash.h>
#include <util/autovector.h>
#include "db/write_batch_internal.h"
#include "db/art/nvm_manager.h"
#include "db/art/nvm_node.h"
#include "db/art/utils.h"
#include "db/art/global_memtable.h"
#include "db/art/node_allocator.h"

namespace ROCKSDB_NAMESPACE {

int SearchVptr(
    InnerNode* inner_node, uint8_t hash, int rows, uint64_t vptr) {

  int buffer_size = GET_NODE_BUFFER_SIZE(inner_node->status_);
  for (int i = 0; i < buffer_size; ++i) {
    if (inner_node->buffer_[i * 2 + 1] == vptr) {
      return -i - 2;
    }
  }

  auto nvm_node = inner_node->nvm_node_;
  auto data = nvm_node->data;
  auto fingerprints = nvm_node->meta.fingerprints_;

  int search_rows = (rows + 1) / 2 - 1;
  int size = rows * 16;

  __m256i target = _mm256_set1_epi8(hash);
  for (int i = search_rows; i >= 0; --i) {
    int base = i << 5;
    __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(fingerprints + base));
    __m256i r = _mm256_cmpeq_epi8(f, target);
    auto res = (unsigned int)_mm256_movemask_epi8(r);
    while (res > 0) {
      int found = 31 - __builtin_clz(res);
      res -= (1 << found);
      int index = found + base;
      if (index < size && vptr == data[index * 2 + 1]) {
        return index;
      }
    }
  }

  return -1;
}

/////////////////////////////////////////////////////////

auto RecordPrefixSize = WriteBatchInternal::kRecordPrefixSize;

void VLogManager::PopFreeSegment() {
  cur_segment_ = GetSegmentFromFreeQueue();
  header_ = (VLogSegmentHeader*)cur_segment_;
  segment_remain_ = vlog_segment_size_ - header_->offset_;
  assert(header_->offset_ < vlog_segment_size_);
  if (!gc_ && free_segments_.size() < force_gc_ratio_) {
    gc_cv_.Signal();
  }
}

char* VLogManager::GetSegmentFromFreeQueue() {
  char* segment = free_segments_.pop_front();
  auto header = (VLogSegmentHeader*)segment;
  std::lock_guard<SpinMutex> lk(header->lock.mutex_);
  header->status_ = kSegmentWriting;
  return segment;
}

void VLogManager::PushToUsedQueue(char* segment) {
  auto header = (VLogSegmentHeader*)segment;
  std::lock_guard<SpinMutex> gc_lk(header->lock.mutex_);
  header->status_ = kSegmentWritten;
  used_segments_.emplace_back(segment);
}

// Why size of vlog header equals vlog_segment_size_ / 128 ?
// We assume that minimum length of single record is 16 byte,
VLogManager::VLogManager(const DBOptions& options)
    : gc_mu_(),
      gc_cv_(&gc_mu_),
      thread_stop_(false),
      vlog_file_size_(options.vlog_file_size),
      vlog_segment_size_(options.vlog_segment_size),
      vlog_segment_num_(vlog_file_size_ / vlog_segment_size_),
      vlog_header_size_(vlog_segment_size_ / 128),
      vlog_bitmap_size_(vlog_header_size_ - sizeof(VLogSegmentHeader)),
      force_gc_ratio_((size_t)
                          (options.vlog_force_gc_ratio_ * vlog_segment_num_)){
  pmemptr_ = GetMappedAddress("vlog");
  char* cur_ptr = pmemptr_;
  for (size_t i = 0; i < vlog_segment_num_; ++i) {
    auto header = new (cur_ptr) VLogSegmentHeader();
    header->status_ = kSegmentFree;
    header->offset_ = vlog_header_size_;
    header->total_count_ = header->compacted_count_ = 0;
    memset(header->bitmap_, -1, vlog_bitmap_size_);
    FLUSH(cur_ptr, vlog_header_size_);
    free_segments_.emplace_back(cur_ptr);
    cur_ptr += vlog_segment_size_;
  }

  MEMORY_BARRIER;

  segment_for_gc_ = GetSegmentFromFreeQueue();

  PopFreeSegment();

  gc_thread_ = std::thread(&VLogManager::BGWorkGarbageCollection, this);
}

VLogManager::~VLogManager() {
  thread_stop_ = true;
  gc_cv_.Signal();
  gc_thread_.join();
}

void VLogManager::SetMemtable(GlobalMemtable* mem) {
  mem_ = mem;
}

uint64_t VLogManager::AddRecord(const Slice& slice, uint32_t record_count) {
  assert(header_->offset_ < vlog_segment_size_);

  uint32_t left = slice.size();
  while (segment_remain_ < left) {
    PushToUsedQueue(cur_segment_);
    PopFreeSegment();
  }

  auto offset = header_->offset_;
  uint64_t vptr = (cur_segment_ - pmemptr_) + offset;
  MEMCPY(cur_segment_ + offset, slice.data(), left, PMEM_F_MEM_NONTEMPORAL);

  segment_remain_ -= left;
  header_->total_count_ += record_count;
  header_->offset_ += left;
  PERSIST(header_, 32);

  assert(segment_remain_ < vlog_segment_size_);

  return vptr;
}

RecordIndex VLogManager::GetFirstIndex(size_t wal_size) const {
  return segment_remain_ < wal_size ? 0 : header_->total_count_;
}

void VLogManager::GetKey(uint64_t vptr, std::string& key) {
  GetActualVptr(vptr);
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, key);
}

void VLogManager::GetKey(uint64_t vptr, Slice& key) {
  GetActualVptr(vptr);
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, &key);
}

ValueType VLogManager::GetKeyValue(uint64_t vptr,
                                   std::string& key, std::string& value,
                                   SequenceNumber& seq_num,
                                   RecordIndex& index) {
  GetActualVptr(vptr);
  ValueType type = ((ValueType*)(pmemptr_ + vptr))[0];
  assert(type == kTypeValue || type == kTypeDeletion);
  seq_num = ((uint64_t*)(pmemptr_ + vptr + 1))[0];
  index = ((uint32_t*)(pmemptr_ + vptr + 9))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, key);

  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, value);
      break;
    default:
      break;
  }

  return type;
}

ValueType VLogManager::GetKeyValue(uint64_t vptr,
                                   std::string& key, std::string& value,
                                   SequenceNumber& seq_num) {
  GetActualVptr(vptr);
  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  assert(type == kTypeValue || type == kTypeDeletion);
  seq_num = ((uint64_t *)(pmemptr_ + vptr + 1))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, key);

  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, value);
      break;
    default:
      break;
  }

  return type;
}

ValueType VLogManager::GetKeyValue(uint64_t vptr,
                                   std::string& key, std::string& value) {
  GetActualVptr(vptr);

  char* segment = pmemptr_ + vptr / vlog_segment_size_ * vlog_segment_size_;
  auto header = (VLogSegmentHeader*)segment;
  assert(header->status_ != kSegmentGC);

  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  assert(type == kTypeValue || type == kTypeDeletion);
  RecordIndex index = ((uint32_t*)(pmemptr_ + vptr + 9))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, key);

  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, value);
      break;
    default:
      break;
  }

  return type;
}

std::vector<GCData> VLogManager::ReadAndSortData(char* segment) {
  auto header = (VLogSegmentHeader*)segment;
  std::vector<GCData> gc_data;
  Slice slice(segment + vlog_header_size_,
              vlog_segment_size_ - vlog_header_size_);

  uint16_t read = 0;
  uint32_t key_length, value_length, key_start;

  KVStruct dummy_info{0, 0};

  auto total_count = header->total_count_;
  auto bitmap = header->bitmap_;
  while (read < total_count) {
    auto record_start = slice.data();
    ValueType type = *((ValueType*)record_start);
    slice.remove_prefix(RecordPrefixSize);

    GetVarint32(&slice, &key_length);
    key_start = slice.data() - record_start;
    slice.remove_prefix(key_length);
    if (type == kTypeValue) {
      GetVarint32(&slice, &value_length);
      slice.remove_prefix(value_length);
    }

    // Not fully tested.
    //if (true) {
    if (bitmap[read / 8] & (1 << (read % 8))) {
      std::string record(record_start, slice.data() - record_start);
      dummy_info.vptr_ = record_start - pmemptr_;
      dummy_info.kv_size_ = record.size() - RecordPrefixSize;
      gc_data.emplace_back(
          key_start, key_length, dummy_info.vptr_, record);
    }
    ++read;
  }

  std::sort(gc_data.begin(), gc_data.end(),
            [](GCData& l, GCData& r) {
              return l.key_ < r.key_;
            });

  return gc_data;
}

void VLogManager::BGWorkGarbageCollection() {
  while (!thread_stop_) {
    gc_cv_.Wait();
    gc_ = true;
    if (thread_stop_) {
      break;
    }

    char* segment_gc = ChooseSegmentToGC();
    auto* header_gc = (VLogSegmentHeader*)segment_gc;

    // If no segment is chosen, wait for next iteration.
    if (!segment_gc) {
      continue;
    }

    std::lock_guard<SpinMutex> gc_lock(header_gc->lock.mutex_);

    std::vector<GCData> gc_data = ReadAndSortData(segment_gc);

    size_t index = 0;
    size_t level;
    uint64_t new_vptr;
    bool stored_in_nvm;

    auto data_count = gc_data.size();

    while (index < data_count) {
      auto& cur_data = gc_data[index];
      auto inner_node = mem_->FindInnerNodeByKey(
          cur_data.key_, level, stored_in_nvm);

      // Right now we don't reclaim free nodes.
      /*if (unlikely(!inner_node)) {
        ++index;
        continue;
      }*/
      assert(inner_node);

      if (!stored_in_nvm) {
        inner_node->opt_lock_.UpgradeToWriteLock();
        while (++index < data_count && gc_data[index].key_ == cur_data.key_) {
          cur_data = gc_data[index];
          if (inner_node->vptr_ == cur_data.vptr_) {
            WriteToNewSegment(cur_data.record_, new_vptr);
            inner_node->vptr_ = new_vptr;
          }
        }
        inner_node->opt_lock_.WriteUnlock(true);
        continue;
      }

      {
        std::string cur_prefix = cur_data.key_.substr(0, level);

        std::lock_guard<std::mutex> flush_lk(inner_node->flush_mutex_);
        if (unlikely(!IS_LEAF(inner_node->status_))) {
          continue;
        }
        auto nvm_node = inner_node->nvm_node_;
        int rows = GET_ROWS(nvm_node->meta.header);

        while (index < data_count) {
          cur_data = gc_data[index];
          if (cur_data.key_.compare(0, level, cur_prefix) != 0) {
            break;
          }

          auto& key = cur_data.key_;
          auto hash = (uint8_t)Hash(key.data(), key.size(), 397);
          auto found_index = SearchVptr(
              inner_node, hash, rows, cur_data.vptr_);
          if (found_index != -1) {
            WriteToNewSegment(cur_data.record_, new_vptr);
            UpdateActualVptr(cur_data.vptr_, new_vptr);

            if (found_index >= 0) {
              nvm_node->data[found_index * 2 + 1] = new_vptr;
            } else {
              inner_node->buffer_[(-found_index - 2) * 2 + 1] = new_vptr;
            }
          }
          ++index;
        }
      }
    }

    header_gc->offset_ = vlog_header_size_;
    header_gc->total_count_ = header_gc->compacted_count_ = 0;
    memset(header_gc->bitmap_, -1, vlog_bitmap_size_);
    // pmem_persist(header, VLogHeaderSize);

    gc_pages_.emplace_back((char*)header_gc);
    header_gc->status_ = kSegmentGC;
    gc_ = false;
  }
}

char* VLogManager::ChooseSegmentToGC() {
  auto num_used = free_segments_.size();
  bool forceGC = num_used < force_gc_ratio_;

  char* segment = nullptr;
  for (size_t i = 0; i < num_used; ++i) {
    segment = used_segments_.pop_front();
    auto header = (VLogSegmentHeader*)segment;
    float compacted_ratio =
        (float)header->compacted_count_ / (float)header->total_count_;
    if (forceGC || compacted_ratio > compacted_ratio_threshold_) {
      break;
    }
    used_segments_.emplace_back(segment);
    segment = nullptr;
  }

  return segment;
}

void VLogManager::WriteToNewSegment(std::string& record, uint64_t& new_vptr) {
  uint32_t left = record.size();
  auto header = (VLogSegmentHeader*)segment_for_gc_;
  auto offset = header->offset_;
  auto count = header->total_count_;
  size_t remain = vlog_segment_size_ - offset;
  if (remain < left) {
    PushToUsedQueue(segment_for_gc_);
    segment_for_gc_ = GetSegmentFromFreeQueue();

    header = (VLogSegmentHeader*)segment_for_gc_;
    offset = header->offset_;
    count = header->total_count_;
    remain = vlog_segment_size_ - offset;
  }

  assert(offset < vlog_segment_size_);
  *(RecordIndex*)(record.data() + 9) = count;
  MEMCPY(segment_for_gc_ + offset, record.data(), left, PMEM_F_MEM_NONTEMPORAL);

  new_vptr = (segment_for_gc_ - pmemptr_) + offset;
  ++header->total_count_;
  header->offset_ += left;
  PERSIST(header, CACHE_LINE_SIZE);
}

void VLogManager::FreeQueue() {
  size_t size = gc_pages_.size();
  while (size--) {
    auto segment = gc_pages_.pop_front();
    free_segments_.emplace_back(segment);
  }
}

}  // namespace ROCKSDB_NAMESPACE
