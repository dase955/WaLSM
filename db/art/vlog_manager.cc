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

namespace ROCKSDB_NAMESPACE {

VLogManager& GetManager() {
  static VLogManager manager;
  return manager;
}

void *OpenVLogFile(int64_t totalSize) {
  int fd = open("/tmp/vlog", O_RDWR|O_CREAT, 00777);
  assert(-1 != fd);

  posix_fallocate(fd, 0, totalSize);
  // lseek(fd, totalSize - 1, SEEK_END);
  // write(fd, "", 1);

  auto addr = (char *)mmap(NULL, totalSize, PROT_READ | PROT_WRITE,MAP_SHARED, fd, 0);
  close(fd);
  return addr;
}

void VLogManager::RecoverOnRestart() {

}

void VLogManager::PopSegment() {
  cur_segment_ = free_pages_.pop_front();
  header_ = (VLogSegmentHeader *)cur_segment_;
  header_->segment_status_ = SegmentStatus::kSegmentWriting;
  offset_ = header_->offset_;
  segment_remain_ = VLogSegmentSize - offset_;
  if (num_free_pages_.fetch_sub(1, std::memory_order_relaxed) < 1 && !gc_) {
    gc_ = true;
  }
}

VLogManager::VLogManager(bool need_recovery) {
  pmemptr_ = (char *) OpenVLogFile(VLogFileSize);

  if (need_recovery) {
    RecoverOnRestart();
    return;
  }

  char *cur_ptr = pmemptr_;
  num_free_pages_.store(VLogSegmentNum, std::memory_order_relaxed);
  for (int i = 0; i < VLogSegmentNum; ++i) {
    auto header = (VLogSegmentHeader *)cur_ptr;
    header->offset_ = SEG_HDR_SIZE;
    header->total_count_ = header->compacted_count_ = 0;
    // pmem_persist(cur_ptr, 64);
    free_pages_.emplace_back(cur_ptr);
    cur_ptr += VLogSegmentSize;
  }

  PopSegment();
}

uint64_t VLogManager::AddRecord(const Slice &slice, uint32_t record_count) {
  // TODO: fix corner case when slice.size_() > VLogSegmentSize
  std::lock_guard<std::mutex> lk(log_mutex_);

  auto left = (int64_t)slice.size();
  if (segment_remain_ < left) {
    used_pages_.emplace_back(cur_segment_);
    PopSegment();
  }

  auto vptr = (cur_segment_ - pmemptr_) + offset_;
  memcpy(cur_segment_ + offset_, slice.data(), left);
  // pmem_memcpy (pmemptr + offset_, slice.data(), left, PMEM_F_MEM_NONTEMPORAL);

  offset_ += left;
  segment_remain_ -= left;

  header_->total_count_ += record_count;
  header_->offset_ = offset_;
  // pmem_persist((void *)header, 32);

  return vptr;
}

void VLogManager::GetKey(uint64_t offset, std::string &key) {
  offset &= 0x0000ffffffffffff;
  Slice slice(pmemptr_ + offset + 9, VLogSegmentSize);
  GetLengthPrefixedSlice(&slice, key);
}

void VLogManager::GetKey(uint64_t offset, Slice &key) {
  offset &= 0x0000ffffffffffff;
  Slice slice(pmemptr_ + offset + 9, VLogSegmentSize);
  GetLengthPrefixedSlice(&slice, &key);
}

ValueType VLogManager::GetKeyValue(uint64_t offset, std::string &key,
                                   std::string &value, SequenceNumber &seq_num) {
  offset &= 0x0000ffffffffffff;
  ValueType type = ((ValueType *)(pmemptr_ + offset))[0];
  seq_num = ((uint64_t *)(pmemptr_ + offset + 1))[0];
  Slice slice(pmemptr_ + offset + 9, VLogSegmentSize);
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

void VLogManager::BGWorkGarbageCollection() {
  auto num_used = used_pages_.size();
  bool forceGC = num_used > VLogSegmentNum * ForceGCThreshold;
  char *segment;
  VLogSegmentHeader *header;
  for (size_t i = 0; i < num_used; ++i) {
    segment = used_pages_.pop_front();
    header = (VLogSegmentHeader *)segment;
    float compactedRatio = (float)header->compacted_count_ / (float)header->total_count_;
    if (forceGC || compactedRatio > CompactedRatioThreshold) {
      break;
    }
    used_pages_.emplace_back(segment);
  }

  std::string key, value;
  std::vector<std::pair<std::string, uint64_t>> data;
  Slice slice(segment + SEG_HDR_SIZE, VLogSegmentSize - SEG_HDR_SIZE);

  uint16_t read = 0;
  auto total_count = header->total_count_;
  while (read++ < total_count) {
    ValueType type = *((ValueType *)slice.data());
    uint64_t vptr = slice.data() - pmemptr_;
    slice.remove_prefix(9);
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
    data.emplace_back(key, vptr);
  }

  std::sort(data.begin(), data.end(),
            [](std::pair<std::string, uint64_t>& l,
               std::pair<std::string, uint64_t>&r) {
    return l.first < r.first;
  });


  header->segment_status_ = SegmentStatus::kSegmentUnused;
  free_pages_.emplace_back(segment);
  num_free_pages_.fetch_add(1, std::memory_order_relaxed);
  gc_ = false;
}

}  // namespace ROCKSDB_NAMESPACE
