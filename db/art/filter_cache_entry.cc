#include "filter_cache_entry.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

#include "db/table_cache.h"
#include "rocksdb/options.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/cachable_entry.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "table/format.h"
#include "table/table_reader.h"

namespace ROCKSDB_NAMESPACE {

// 构造函数，可以初始化成员变量
// TODO pass right parameters
FilterCacheEntry::FilterCacheEntry(
    const uint32_t segment_id, const BlockBasedTable* table,
    FilterCache* filter_cache, const std::vector<BlockHandle>& block_handles) {
  segment_id_ = segment_id;
  table_ = table;
  filter_cache_ = filter_cache;
  loaded_units_num_ = 0;

  // fill block_handles from the input vector, or fill with null handles
  assert(block_handles.size() == MAX_UNITS_NUM);
  block_handles_.fill(BlockHandle::NullBlockHandle());
  cache_handles_.fill(nullptr);
  for (size_t i = 0; i < block_handles.size(); i++) {
    block_handles_[i] = block_handles[i];
  }

  // load units into memory, then only modify loaded_units_num_
  prefetch_units();
}

// 清理成员变量，避免内存泄漏，如果new了空间，就可能需要在这里清理
FilterCacheEntry::~FilterCacheEntry() {}

size_t FilterCacheEntry::approximate_size() {
  uint32_t sum = 0;
  // for (size_t i = 0; i < loaded_units_num_; i++) {
  //   if (cache_handles_[i] == nullptr) {
  //     continue;
  //   }
  //   sum += cache_handles_[i]->value_->ApproximateMemoryUsage();
  // }
  // sum *= 8;  // convert to bits
  sum += DEFAULT_UNIT_SIZE * loaded_units_num_;
  return sum;
}

std::vector<CachableEntry<ParsedFullFilterBlock>>
FilterCacheEntry::get_filter_blocks() {
  uint32_t units_num = std::max(loaded_units_num_, uint32_t(1));
  assert(units_num > 0);

  rwlock.ReadLock();
  std::vector<CachableEntry<ParsedFullFilterBlock>> result;

  result.reserve(units_num);
  for (size_t i = 0; i < units_num; i++) {
    if (UNLIKELY(cache_handles_[i] == nullptr)) {
      result.emplace_back(nullptr, nullptr, nullptr, false);
      result[i].Reset();
      continue;
    }
    result.emplace_back(cache_handles_[i]->value_.get(), filter_cache_,
                        cache_handles_[i].get(), false);
  }
  rwlock.ReadUnlock();
  return result;
}

void FilterCacheEntry::enable_units(uint32_t target_unit_num) {
  if (target_unit_num > MAX_UNITS_NUM) {
    target_unit_num = MAX_UNITS_NUM;
  }

  // std::cout << "segment id: " << segment_id_ << ", enable units num from " << loaded_units_num_ << " to " << target_unit_num << std::endl;

  rwlock.WriteLock();
  loaded_units_num_ = target_unit_num;
  rwlock.WriteUnlock();

  // static std::atomic<uint32_t> target_unit_num_5_counter{0};
  // static std::atomic<uint32_t> target_unit_num_12_counter{0};
  // if (target_unit_num == 5) {
  //   target_unit_num_5_counter.fetch_add(1);
  // } else if (target_unit_num == 12) {
  //   target_unit_num_12_counter.fetch_add(1);
  // }
  // std::cout << "target_unit_num_5_counter: " << target_unit_num_5_counter.load() << ", target_unit_num_12_counter: " << target_unit_num_12_counter.load() << std::endl;
}

void FilterCacheEntry::prefetch_units() {
  uint32_t target_unit_num = MAX_UNITS_NUM;
  uint32_t prefetch_success_num = 0;

  rwlock.WriteLock();
  const ReadOptions read_options;
  for (uint32_t i = 0; i < target_unit_num; i++) {
    // do nothing for null block handle
    if (block_handles_[i] == BlockHandle::NullBlockHandle()) {
      continue;
    }
    CachableEntry<ParsedFullFilterBlock> block_entry;
    Status s = table_->RetrieveBlock(
        nullptr, read_options, block_handles_[i],
        UncompressionDict::GetEmptyDict(), &block_entry, BlockType::kFilter,
        nullptr, nullptr,
        /* for_compaction */ false, /* use_cache */ false);

    
    if (s.ok()) {
      prefetch_success_num++;
    }

    // do nothing if no data retrieved
    if (!s.ok()) {
      std::cout << "failed to retrive filter data, segment id: " << segment_id_
                << std::endl;
      cache_handles_[i].reset();
      units_[i].reset();
      break;
    }

    units_[i] =
        std::shared_ptr<ParsedFullFilterBlock>(block_entry.ReleaseValue());
    cache_handles_[i] =
        // std::make_shared<FilterCacheDataHandle>(units_[i], filter_cache_);
        std::shared_ptr<FilterCacheDataHandle>(
            new FilterCacheDataHandle(units_[i], filter_cache_));
  }
  rwlock.WriteUnlock();

  // std::cout << "segment id: " << segment_id_ << ", prefetch success num: " << prefetch_success_num << std::endl;
  // std::cout << "used bytes:" ;
  // for (uint32_t i = 0; i < target_unit_num; i++) {
  //   if (cache_handles_[i] == nullptr) {
  //     std::cout << "null ";
  //     continue;
  //   }
  //   std::cout << cache_handles_[i]->value_->ApproximateMemoryUsage() << " ";
  // }
  // std::cout << std::endl;
}

FilterCacheEntry::FilterCacheDataHandle::FilterCacheDataHandle(
    DataPtr value, FilterCache* cache)
    : value_(value), cache_(cache) {}
}  // namespace ROCKSDB_NAMESPACE