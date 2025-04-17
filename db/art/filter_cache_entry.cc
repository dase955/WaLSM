#include "filter_cache_entry.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
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
FilterCacheEntry::FilterCacheEntry(const uint32_t segment_id,
                                   const BlockBasedTable* table,
                                   FilterCache* filter_cache,
                                   const std::vector<BlockHandle>& block_handles) {
  segment_id_ = segment_id;
  table_ = table;
  filter_cache_ = filter_cache;
  loaded_units_num_ = 0;

  // fill block_handles from the input vector, or fill with null handles
  assert(block_handles.size() <= MAX_UNITS_NUM);
  block_handles_.fill(BlockHandle::NullBlockHandle());
  cache_handles_.fill(nullptr);
  for (size_t i = 0; i < block_handles.size(); i++) {
    block_handles_[i] = block_handles[i];
  }
}

// 清理成员变量，避免内存泄漏，如果new了空间，就可能需要在这里清理
FilterCacheEntry::~FilterCacheEntry() {}

size_t FilterCacheEntry::approximate_size() {
  uint32_t sum = 0;
  for (size_t i = 0; i < loaded_units_num_; i++) {
    if (cache_handles_[i] == nullptr) {
      continue;
    }
    sum += cache_handles_[i]->value_->ApproximateMemoryUsage();
  }
  sum *= 8;  // convert to bits
  return sum;
}

std::vector<CachableEntry<ParsedFullFilterBlock>>
FilterCacheEntry::get_filter_blocks() {
  rwlock.ReadLock();
  std::vector<CachableEntry<ParsedFullFilterBlock>> result;

  if (loaded_units_num_ == 0) {
    rwlock.ReadUnlock();
    return result;
  }

  result.reserve(loaded_units_num_);
  for (size_t i = 0; i < loaded_units_num_; i++) {
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

  rwlock.WriteLock();
  if (target_unit_num < loaded_units_num_) {
    for (uint32_t i = target_unit_num; i < loaded_units_num_; i++) {
      cache_handles_[i].reset();
      units_[i].reset();
    }
    loaded_units_num_ = target_unit_num;
  } else if (target_unit_num > loaded_units_num_) {
    const ReadOptions read_options;
    for (uint32_t i = loaded_units_num_; i < target_unit_num; i++) {
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

      // do nothing if no data retrieved
      if (!s.ok()) {
        break;
      }

      units_[i] =
          std::shared_ptr<ParsedFullFilterBlock>(block_entry.ReleaseValue());
      cache_handles_[i] =
          // std::make_shared<FilterCacheDataHandle>(units_[i], filter_cache_);
          std::shared_ptr<FilterCacheDataHandle>(
              new FilterCacheDataHandle(units_[i], filter_cache_));
    }
    loaded_units_num_ = target_unit_num;
  }
  rwlock.WriteUnlock();
}

FilterCacheEntry::FilterCacheDataHandle::FilterCacheDataHandle(
    DataPtr value, FilterCache* cache)
    : value_(value), cache_(cache) {}
}  // namespace ROCKSDB_NAMESPACE