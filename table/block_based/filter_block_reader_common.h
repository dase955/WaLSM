//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#pragma once

#include <cassert>
#include "table/block_based/cachable_entry.h"
#include "table/block_based/filter_block.h"

namespace ROCKSDB_NAMESPACE {

class BlockBasedTable;
class FilePrefetchBuffer;

// Encapsulates common functionality for the various filter block reader
// implementations. Provides access to the filter block regardless of whether
// it is owned by the reader or stored in the cache, or whether it is pinned
// in the cache or not.
template <typename TBlocklike>
class FilterBlockReaderCommon : public FilterBlockReader {
 public:
  // init method
  FilterBlockReaderCommon(const BlockBasedTable* t,
                          CachableEntry<TBlocklike>&& filter_block)
      : table_(t), filter_block_(std::move(filter_block)) {
    assert(table_);
  }

 protected:
  // input table, call table->RetrieveBlock() and return status
  static Status ReadFilterBlock(const BlockBasedTable* table,
                                FilePrefetchBuffer* prefetch_buffer,
                                const ReadOptions& read_options, bool use_cache,
                                GetContext* get_context,
                                BlockCacheLookupContext* lookup_context,
                                CachableEntry<TBlocklike>* filter_block);
  // return table_
  const BlockBasedTable* table() const { return table_; }
  // not used in our work
  const SliceTransform* table_prefix_extractor() const;
  // return table_->get_rep()->whole_key_filtering;
  bool whole_key_filtering() const;
  // return table_->get_rep()->table_options.cache_index_and_filter_blocks;
  // return true when cache filter in block cache
  // should return false in WaLSM+, we will design new cache space for filter
  bool cache_filter_blocks() const;
  // if cached, because filter_block_ already own the cache value
  // we only need to call filter_block->SetUnownedValue(filter_block_.GetValue());
  // so filter_block only have the reference of cached entry
  // if not cached, we read from block, load into filter block(owned) 
  // and return status.
  Status GetOrReadFilterBlock(bool no_io, GetContext* get_context,
                              BlockCacheLookupContext* lookup_context,
                              CachableEntry<TBlocklike>* filter_block) const;
  // if GetOwnValue() = true，return Usage，otherwise 0
  size_t ApproximateFilterBlockMemoryUsage() const;

 private:
  const BlockBasedTable* table_;
  CachableEntry<TBlocklike> filter_block_;
};

}  // namespace ROCKSDB_NAMESPACE
