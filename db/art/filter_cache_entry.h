#pragma once

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "macros.h"
#include "port/port_posix.h"
#include "table/block_based/cachable_entry.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "table/format.h"

namespace ROCKSDB_NAMESPACE {
class TableCache;  // forward declaration
class BlockBasedTable;
class FilterCache;

// 先在filter
// cache里为每个segment默认启用总bits-per-key=8，随着写入的segment的增加，
// 一旦已经占用了filter cache最大容量的一定阈值(如80%),
// 就利用GreedyAlgo计算规划问题，并进行模型训练 一旦filter
// cache已满，就进入filter cache的double
// heap调整，我们只需将新的segment用模型进行预测
// 将新segment的node插入到两个heap里，在后台启动一个线程，自行调整两个堆，并不断返回调整的结果
// 得到结果后，我们可以立即对filter
// units的启用情况进行调节，也可以先保存后面批量调整 具体见文档

// 注意加上一些必要的英文注释
// filter cache主要为一个map, key是segment id(uint32_t),
// value就为FilterCacheItem类 成员函数需要在filter_cache_item.cc里定义

// TODO: how to get block_handles?
class FilterCacheEntry {
  using DataPtr = std::shared_ptr<ParsedFullFilterBlock>;

 private:
  const BlockBasedTable* table_;
  FilterCache* filter_cache_;

  struct FilterCacheDataHandle : public Cache::Handle {
    DataPtr value_;
    FilterCache* cache_;

    FilterCacheDataHandle(DataPtr value, FilterCache* cache);
  };
  using HandlePtr = std::shared_ptr<FilterCacheDataHandle>;

  uint32_t segment_id_;
  uint32_t loaded_units_num_;
  std::array<DataPtr, MAX_UNITS_NUM> units_{};
  std::array<BlockHandle, MAX_UNITS_NUM> block_handles_{};
  std::array<HandlePtr, MAX_UNITS_NUM> cache_handles_{};
  mutable port::RWMutex rwlock;

 public:
  // 构造函数，可以初始化成员变量
  // TODO pass right parameters
  FilterCacheEntry(const uint32_t segment_id, const BlockBasedTable* table,
                   FilterCache* filter_cache, const std::vector<BlockHandle>& block_handles);

  // 清理成员变量，避免内存泄漏，如果new了空间，就可能需要在这里清理
  ~FilterCacheEntry();

  // 占用的内存空间，这里估计总共使用的filter units占用的空间就行了
  // 注意，返回的空间大小为占用的bits数量，不是bytes数量
  size_t approximate_size();

  // 根据目前已经启用的units数，启用或禁用filter units
  // 输入需要启用的units数，决定启用、禁用还是不处理
  // units_num : [MIN_UNITS_NUM, MAX_UNITS_NUM]
  void enable_units(const uint32_t units_num);

  // 获取缓存的 filter Block
  std::vector<CachableEntry<ParsedFullFilterBlock>> get_filter_blocks();
};
}  // namespace ROCKSDB_NAMESPACE