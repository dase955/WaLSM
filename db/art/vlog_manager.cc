//
// Created by joechen on 2022/4/3.
//

#include "vlog_manager.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <cassert>
#include <csignal>
#include <string>
#include <cstring>

#include "db/art/nvm_node.h"
#include "db/art/utils.h"
#include "db/art/global_memtable.h"
#include "db/art/node_allocator.h"
#include "util/autovector.h"

namespace ROCKSDB_NAMESPACE {

char* OpenVLogFile(uint64_t total_size) {
  int fd = open("/tmp/vlog", O_RDWR|O_CREAT, 00777);
  assert(-1 != fd);

  //posix_fallocate(fd, 0, total_size);
  lseek(fd, total_size - 1, SEEK_END);
  write(fd, "", 1);

  char* ptr = (char *)mmap(nullptr, total_size, PROT_READ | PROT_WRITE,MAP_SHARED, fd, 0);
  close(fd);
  return ptr;
}

void VLogManager::RecoverOnRestart() {

}

void VLogManager::PopFreeSegment() {
  cur_segment_ = free_pages_.pop_front();
  header_ = (VLogSegmentHeader*)cur_segment_;
  offset_ = header_->offset_;
  count_in_segment_ = header_->total_count_;
  segment_remain_ = VLogSegmentSize - offset_;
  int num_free_pages = free_pages_.size();
  if (!gc_ && num_free_pages < ForceGCThreshold) {
    gc_ = true;
    gc_cv_.Signal();
  }
}

// Why size of vlog header equals vlog_segment_size_ / 128 ?
// We assume that minimum length of single record is 16 byte,
VLogManager::VLogManager(const DBOptions& options)
    : gc_mu_(), gc_cv_(&gc_mu_), thread_stop_(false) {
  pmemptr_ = OpenVLogFile(VLogFileSize);

  char* cur_ptr = pmemptr_;
  for (int i = 0; i < VLogSegmentNum; ++i) {
    auto header = new (cur_ptr) VLogSegmentHeader();
    header->offset_ = VLogHeaderSize;
    header->total_count_ = header->compacted_count_ = 0;
    memset(header->bitmap_, -1, VLogBitmapSize);
    // pmem_persist(cur_ptr, VLogHeaderSize);
    free_pages_.emplace_back(cur_ptr);
    cur_ptr += VLogSegmentSize;
  }

  PopFreeSegment();

  gc_thread_ = std::thread(&VLogManager::BGWorkGarbageCollection, this);
}

VLogManager::~VLogManager() {
  thread_stop_ = true;
  gc_cv_.Signal();
  gc_thread_.join();
  munmap(pmemptr_, VLogFileSize);
}

void VLogManager::SetMemtable(GlobalMemtable* mem) {
  mem_ = mem;
}

uint64_t VLogManager::AddRecord(const Slice& slice, uint32_t record_count) {
  // std::lock_guard<std::mutex> log_lk(log_mutex_);

  auto left = (int64_t)slice.size();
  if (segment_remain_ < left) {
    // TODO: new segment may still doesn't have enough space for record.
    used_pages_.emplace_back(cur_segment_);
    PopFreeSegment();
  }

  auto vptr = (cur_segment_ - pmemptr_) + offset_;
  memcpy(cur_segment_ + offset_, slice.data(), left);
  // pmem_memcpy (pmemptr + offset_, slice.data(), left, PMEM_F_MEM_NONTEMPORAL);

  offset_ += left;
  segment_remain_ -= left;
  count_in_segment_ += record_count;

  header_->total_count_ = count_in_segment_;
  header_->offset_ = offset_;
  // pmem_persist((void*)header, 32);

  return vptr;
}

RecordIndex VLogManager::GetFirstIndex(size_t wal_size) const {
  return segment_remain_ < wal_size ? 0 : count_in_segment_;
}

void VLogManager::GetKey(uint64_t vptr, std::string &key) {
  GetActualVptr(vptr);
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, VLogSegmentSize);
  GetLengthPrefixedSlice(&slice, key);
}

void VLogManager::GetKey(uint64_t vptr, Slice &key) {
  GetActualVptr(vptr);
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, VLogSegmentSize);
  GetLengthPrefixedSlice(&slice, &key);
}

ValueType VLogManager::GetKeyValue(
    uint64_t vptr, std::string& key, std::string& value,
    SequenceNumber& seq_num, RecordIndex &index) {
  GetActualVptr(vptr);
  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  seq_num = ((uint64_t *)(pmemptr_ + vptr + 1))[0];
  index = ((uint32_t *)(pmemptr_ + vptr + 9))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, VLogSegmentSize);
  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, key);
      GetLengthPrefixedSlice(&slice, value);
      break;
    case kTypeDeletion:
      GetLengthPrefixedSlice(&slice, key);
      break;
    default:
      break;
  }

  return type;
}

ValueType VLogManager::GetKeyValue(uint64_t vptr, std::string& key,
                                   std::string& value, SequenceNumber& seq_num) {
  GetActualVptr(vptr);
  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  seq_num = ((uint64_t *)(pmemptr_ + vptr + 1))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, VLogSegmentSize);
  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, key);
      GetLengthPrefixedSlice(&slice, value);
      break;
    case kTypeDeletion:
      GetLengthPrefixedSlice(&slice, key);
      break;
    default:
      break;
  }

  return type;
}

ValueType VLogManager::GetKeyValue(
    uint64_t vptr, std::string& key, std::string& value) {
  GetActualVptr(vptr);
  ValueType type = ((ValueType *)(pmemptr_ + vptr))[0];
  RecordIndex index = ((uint32_t *)(pmemptr_ + vptr + 9))[0];
  Slice slice(pmemptr_ + vptr + RecordPrefixSize, VLogSegmentSize);
  switch (type) {
    case kTypeValue:
      GetLengthPrefixedSlice(&slice, key);
      GetLengthPrefixedSlice(&slice, value);
      break;
    case kTypeDeletion:
      GetLengthPrefixedSlice(&slice, key);
      break;
    default:
      break;
  }

  return type;
}

int SearchVptr(
    InnerNode* inner_node, uint64_t hash, int rows, uint64_t vptr) {

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

  __m256i target = _mm256_set1_epi8((uint8_t)hash);
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


std::vector<GCData> VLogManager::ReadAndSortData(char* segment) {
  auto header = (VLogSegmentHeader*)segment;
  std::vector<GCData> gc_data;
  Slice slice(segment + VLogHeaderSize, VLogSegmentSize - VLogHeaderSize);

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
    if (thread_stop_) {
      break;
    }

    char* segment_gc = ChooseSegmentToGC();
    auto* header_gc = (VLogSegmentHeader*)segment_gc;

    // If no segment is chosen, wait for next iteration.
    if (!segment_gc) {
      printf("No segment selected.\n");
      continue;
    }

    std::lock_guard<SpinMutex> gc_lock(header_gc->lock.mutex_);
    std::vector<GCData> gc_data = ReadAndSortData(segment_gc);

    auto new_segment = free_pages_.pop_front();

    size_t index = 1;
    size_t level;
    uint64_t new_vptr;
    bool stored_in_nvm;

    auto& cur_data = gc_data.front();
    auto data_count = gc_data.size();

    while (index < data_count) {
      auto inner_node = mem_->FindInnerNodeByKey(cur_data.key_, level, stored_in_nvm);
      if (unlikely(!inner_node)) {
        continue;
      }

      if (!stored_in_nvm) {
        while (index < data_count && gc_data[index].key_ == cur_data.key_) {
          cur_data = gc_data[index++];
        }

        if (inner_node->vptr_ == cur_data.vptr_) {
          new_segment = WriteToNewSegment(
              new_segment, cur_data.record_, new_vptr);
          inner_node->vptr_ = new_vptr;
        }
        cur_data = gc_data[index];
        continue;
      }

      {
        std::string cur_prefix = cur_data.key_.substr(0, level);

        std::lock_guard<std::mutex> flush_lk(inner_node->flush_mutex_);
        //std::lock_guard<std::mutex> gc_lk(inner_node->gc_mutex_);
        if (unlikely(!IS_LEAF(inner_node->status_))) {
          continue;
        }
        auto nvm_node = inner_node->nvm_node_;
        int rows = GET_ROWS(nvm_node->meta.header);

        do {
          auto& key = cur_data.key_;
          auto hash = Hash(key.data(), key.size());
          auto found_index = SearchVptr(
              inner_node, hash, rows, cur_data.vptr_);
          if (found_index != -1) {
            new_segment = WriteToNewSegment(
                new_segment, cur_data.record_, new_vptr);
            UpdateActualVptr(cur_data.vptr_, new_vptr);

            std::string old_key, old_value;
            std::string new_key, new_value;
            if (found_index >= 0) {
#if 0
            GetKeyValue(nvm_node->data[found_index * 2 + 1], old_key, old_value);
            GetKeyValue(new_vptr, new_key, new_value);
            assert(old_key == new_key);
#endif
              nvm_node->data[found_index * 2 + 1] = new_vptr;
            } else {
              GetKeyValue(inner_node->buffer_[(-found_index - 2) * 2 + 1], old_key, old_value);
              GetKeyValue(new_vptr, new_key, new_value);
              inner_node->buffer_[(-found_index - 2) * 2 + 1] = new_vptr;
            }
          }
          cur_data = gc_data[index++];
        } while (index < data_count &&
                 cur_data.key_.compare(0, level, cur_prefix) == 0);
      }
    }

    header_gc->offset_ = VLogSegmentSize;
    header_gc->total_count_ = header_gc->compacted_count_ = 0;
    memset(header_gc->bitmap_, -1, VLogBitmapSize);
    // pmem_persist(header, VLogHeaderSize);

    free_pages_.emplace_back((char*)header_gc);
    free_pages_.emplace_front((char*)new_segment);
    gc_ = false;

    printf("GC done.\n");
  }
}

char* VLogManager::ChooseSegmentToGC() {
  auto num_used = free_pages_.size();
  bool forceGC = num_used < ForceGCThreshold;

  char* segment = nullptr;
  for (size_t i = 0; i < num_used; ++i) {
    segment = used_pages_.pop_front();
    auto header = (VLogSegmentHeader*)segment;
    float compacted_ratio =
        (float)header->compacted_count_ / (float)header->total_count_;
    if (forceGC || compacted_ratio > CompactedRatioThreshold) {
      break;
    }
    used_pages_.emplace_back(segment);
    segment = nullptr;
  }

  return segment;
}

char* VLogManager::WriteToNewSegment(char* segment, std::string& record,
                                     uint64_t& new_vptr) {
  auto header = (VLogSegmentHeader*)segment;
  auto offset = header->offset_;
  uint32_t left = record.size();
  size_t remain = VLogSegmentSize - offset;
  if (remain < left) {
    used_pages_.emplace_back(segment);
    segment = free_pages_.pop_front();
    header = (VLogSegmentHeader*)segment;
    offset = header->offset_;
  }
  auto count = header->total_count_;
  *(RecordIndex*)(record.data() + 9) = count;
  memcpy(segment + offset, record.data(), left);
  // pmem_memcpy (pmemptr + offset_, slice.data(), left, PMEM_F_MEM_NONTEMPORAL);

  new_vptr = (segment - pmemptr_) + offset;
  ++header->total_count_;
  header->offset_ += left;
  // pmem_persist((void*)header, 32);

  return segment;
}

void VLogManager::TestGC() {
  char* segment_gc = used_pages_.pop_front();
  auto* header_gc = (VLogSegmentHeader*)segment_gc;

  std::lock_guard<SpinMutex> gc_lock(header_gc->lock.mutex_);
  std::vector<GCData> gc_data = ReadAndSortData(segment_gc);

  auto new_segment = free_pages_.pop_front();

  size_t index = 1;
  size_t level;
  uint64_t new_vptr;
  bool stored_in_nvm;

  auto& cur_data = gc_data.front();
  auto data_count = gc_data.size();

  while (index < data_count) {
    auto inner_node = mem_->FindInnerNodeByKey(cur_data.key_, level, stored_in_nvm);
    if (unlikely(!inner_node)) {
      continue;
    }

    if (!stored_in_nvm) {
      while (index < data_count && gc_data[index].key_ == cur_data.key_) {
        cur_data = gc_data[index++];
      }

      if (inner_node->vptr_ == cur_data.vptr_) {
        new_segment = WriteToNewSegment(
            new_segment, cur_data.record_, new_vptr);
        inner_node->vptr_ = new_vptr;
      }
      cur_data = gc_data[index];
      continue;
    }

    {
      std::string cur_prefix = cur_data.key_.substr(0, level);

      std::lock_guard<std::mutex> flush_lk(inner_node->flush_mutex_);
      //std::lock_guard<std::mutex> gc_lk(inner_node->gc_mutex_);
      if (unlikely(!IS_LEAF(inner_node->status_))) {
        continue;
      }
      auto nvm_node = inner_node->nvm_node_;
      auto rows = GET_ROWS(nvm_node->meta.header);

      do {
        auto& key = cur_data.key_;
        auto hash = Hash(key.data(), key.size());
        auto found_index = SearchVptr(
            inner_node, hash, rows, cur_data.vptr_);
        if (found_index != -1) {
          new_segment = WriteToNewSegment(
              new_segment, cur_data.record_, new_vptr);
          UpdateActualVptr(cur_data.vptr_, new_vptr);

          std::string old_key, old_value;
          std::string new_key, new_value;
          if (found_index >= 0) {
#if 0
            GetKeyValue(nvm_node->data[found_index * 2 + 1], old_key, old_value);
            GetKeyValue(new_vptr, new_key, new_value);
            assert(old_key == new_key);
#endif
            nvm_node->data[found_index * 2 + 1] = new_vptr;
          } else {
            GetKeyValue(inner_node->buffer_[(-found_index - 2) * 2 + 1], old_key, old_value);
            GetKeyValue(new_vptr, new_key, new_value);
            inner_node->buffer_[(-found_index - 2) * 2 + 1] = new_vptr;
          }
        }
        cur_data = gc_data[index++];
      } while (index < data_count &&
               cur_data.key_.compare(0, level, cur_prefix) == 0);
    }
  }

  header_gc->offset_ = VLogSegmentSize;
  header_gc->total_count_ = header_gc->compacted_count_ = 0;
  memset(header_gc->bitmap_, -1, VLogBitmapSize);
  // pmem_persist(header, VLogHeaderSize);

  free_pages_.emplace_back((char*)header_gc);
  gc_ = false;
}

}  // namespace ROCKSDB_NAMESPACE
