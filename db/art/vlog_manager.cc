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
  PERSIST(segment, 8);
  return segment;
}

void VLogManager::PushToUsedQueue(char* segment) {
  auto header = (VLogSegmentHeader*)segment;
  std::lock_guard<SpinMutex> gc_lk(header->lock.mutex_);
  header->status_ = kSegmentWritten;
  PERSIST(segment, 8);
  used_segments_.emplace_back(segment);
}

// Why size of vlog header equals vlog_segment_size_ / 128 ?
// We assume that minimum length of single record is 16 byte,
VLogManager::VLogManager(const DBOptions& options, bool recovery)
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
  recovery ? Recover() : Initialize();

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

void VLogManager::Recover() {
  char* cur_ptr = pmemptr_;
  for (size_t i = 0; i < vlog_segment_num_; ++i) {
    auto header = (VLogSegmentHeader*)cur_ptr;
    if (header->status_ == kSegmentWriting ||
        header->status_ == kSegmentWritten) {
      header->status_ = kSegmentWritten;
      FLUSH(cur_ptr, vlog_header_size_);
      used_segments_.emplace_back(cur_ptr);
      cur_ptr += vlog_segment_size_;
      continue;
    }

    header->status_ = kSegmentFree;
    header->offset_ = vlog_header_size_;
    header->total_count_ = header->compacted_count_ = 0;
    memset(header->bitmap_, -1, vlog_bitmap_size_);
    FLUSH(cur_ptr, vlog_header_size_);
    free_segments_.emplace_back(cur_ptr);
    cur_ptr += vlog_segment_size_;
  }
}

void VLogManager::Initialize() {
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
}

void VLogManager::SetMemtable(GlobalMemtable* mem) {
  mem_ = mem;
}

uint64_t VLogManager::AddRecord(const Slice& slice, uint32_t record_count) {
  assert(header_->offset_ <= vlog_segment_size_);

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

void VLogManager::GetKeyIndex(uint64_t vptr, std::string &key,
                              RecordIndex& index) {
  GetActualVptr(vptr);
  index = ((uint32_t*)(pmemptr_ + vptr + 9))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, vlog_segment_size_);
  GetLengthPrefixedSlice(&slice, key);
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
                                   std::string& key, std::string& value) {
  GetActualVptr(vptr);

  [[maybe_unused]] char* segment =
      pmemptr_ + vptr / vlog_segment_size_ * vlog_segment_size_;
  assert(((VLogSegmentHeader*)segment)->status_ != kSegmentGC);


  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  assert(type == kTypeValue || type == kTypeDeletion);
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

void VLogManager::ReadAndSortData(std::vector<char*>& segments) {
  int estimate = 0;
  for (auto segment : segments) {
    auto header = (VLogSegmentHeader*)segment;
    estimate += (header->total_count_ - header->compacted_count_);
  }

  gc_data.clear();
  gc_data.reserve(estimate);

  for (auto segment : segments) {
    auto header = (VLogSegmentHeader*)segment;
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
        dummy_info.kv_size_ = record.size();
        gc_data.emplace_back(
            key_start, key_length, dummy_info.vptr_, record);
      }
      ++read;
    }
  }

  std::sort(gc_data.begin(), gc_data.end(),
            [](GCData& l, GCData& r) {
              return l.key_ < r.key_;
            });
}

void VLogManager::BGWorkGarbageCollection() {
  std::vector<char*> segments;

  while (!thread_stop_) {
    gc_cv_.Wait();
    gc_ = true;
    if (thread_stop_) {
      break;
    }

    segments.clear();
    for (int i = 0; i < 4; ++i) {
      char* segment_gc = ChooseSegmentToGC();
      if (!segment_gc) {
        continue;
      }
      segments.push_back(segment_gc);
    }

    if (segments.empty()) {
      continue;
    }

    for (auto segment : segments) {
      auto* header_gc = (VLogSegmentHeader*)segment;
      std::lock_guard<SpinMutex> gc_lock(header_gc->lock.mutex_);
      header_gc->status_ = kSegmentGC;
      PERSIST(segment, 8);
    }

    ReadAndSortData(segments);

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
      if (unlikely(!inner_node)) {
        ++index;
        continue;
      }

      if (!stored_in_nvm) {
        inner_node->opt_lock_.WriteLock();
        std::string& vptr_key = cur_data.key_;
        while (index < data_count && gc_data[index].key_ == vptr_key) {
          cur_data = gc_data[index++];
          if (inner_node->vptr_ == cur_data.vptr_) {
            WriteToNewSegment(cur_data.record_, new_vptr);
            UpdateActualVptr(cur_data.vptr_, new_vptr);
            inner_node->vptr_ = new_vptr;
          }
        }
        inner_node->opt_lock_.WriteUnlock(false);
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

    for (auto segment : segments) {
      auto* header_gc = (VLogSegmentHeader*)segment;
      header_gc->offset_ = vlog_header_size_;
      header_gc->total_count_ = header_gc->compacted_count_ = 0;
      memset(header_gc->bitmap_, -1, vlog_bitmap_size_);
      // pmem_persist(header, VLogHeaderSize);

      gc_pages_.emplace_back((char*)header_gc);
    }

    gc_ = false;
  }
}

char* VLogManager::ChooseSegmentToGC() {
  char* segment = nullptr;

  auto num_used = used_segments_.size();
  for (size_t i = 0; i < num_used; ++i) {
    segment = used_segments_.pop_front();
    auto header = (VLogSegmentHeader*)segment;
    float compacted_ratio =
        (float)header->compacted_count_ / (float)header->total_count_;
    if (compacted_ratio > compacted_ratio_threshold_) {
      break;
    }
    used_segments_.emplace_back(segment);
    segment = nullptr;
  }

  if (!segment) {
    segment = used_segments_.pop_front();
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
    auto header = (VLogSegmentHeader*)segment;
    std::lock_guard<SpinMutex> lk(header->lock.mutex_);
    header->status_ = kSegmentFree;
    PERSIST(segment, 8);
    free_segments_.emplace_back(segment);
  }
}

void VLogManager::UpdateBitmap(
    std::unordered_map<uint64_t, std::vector<RecordIndex>>& all_indexes) {
  for (auto& pair : all_indexes) {
    auto segment_id = pair.first;
    auto& indexes = pair.second;
    assert(segment_id < vlog_segment_num_);

    auto header = (VLogSegmentHeader*)(
        pmemptr_ + vlog_segment_size_ * segment_id);

    if (!header->lock.mutex_.try_lock()) {
      continue;
    }

    if (header->status_ != kSegmentWritten) {
      header->lock.mutex_.unlock();
      continue;
    }

    auto bitmap = header->bitmap_;
    for (auto& index : indexes) {
      assert(index / 8 < vlog_bitmap_size_);
      bitmap[index / 8] &= ~(1 << (index % 8));
    }
    header->compacted_count_ += indexes.size();
    PERSIST(header, vlog_header_size_);

    header->lock.mutex_.unlock();
  }

  MEMORY_BARRIER;
}

void VLogManager::UpdateBitmap([[maybe_unused]] uint64_t vptr) {

}

float VLogManager::Estimate() {
  int total = 0;
  int compacted = 0;
  for (size_t i = 0; i < vlog_segment_num_; ++i) {
    auto header = (VLogSegmentHeader*)(pmemptr_ + vlog_segment_size_ * i);
    if (header->status_ != kSegmentWritten) {
      continue;
    }
    total += header->total_count_;
    compacted += header->compacted_count_;
  }
  return (float)compacted / (float)total;
}

}  // namespace ROCKSDB_NAMESPACE
