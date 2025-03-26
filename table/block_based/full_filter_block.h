//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include "db/dbformat.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "table/block_based/filter_block_reader_common.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {

class FilterPolicy;
class FilterBitsBuilder;
class FilterBitsReader;

// A FullFilterBlockBuilder is used to construct a full filter for a
// particular Table.  It generates a single string which is stored as
// a special block in the Table.
// The format of full filter block is:
// +----------------------------------------------------------------+
// |              full filter for all keys in sst file              |
// +----------------------------------------------------------------+
// The full filter can be very large. At the end of it, we put
// num_probes: how many hash functions are used in bloom filter
//
class FullFilterBlockBuilder : public FilterBlockBuilder {
 public:
  // when format version < 5, use LegacyBloomBitsBuilder
  // In our work, format version = 4
  explicit FullFilterBlockBuilder(const SliceTransform* prefix_extractor,
                                  bool whole_key_filtering,
                                  FilterBitsBuilder* filter_bits_builder);
  // No copying allowed
  FullFilterBlockBuilder(const FullFilterBlockBuilder&) = delete;
  void operator=(const FullFilterBlockBuilder&) = delete;

  // bits_builder is created in filter_policy, it should be passed in here
  // directly. and be deleted here
  ~FullFilterBlockBuilder() {}

  // default return false
  virtual bool IsBlockBased() override { return false; }
  // not implemented in FullFilterBlock
  virtual void StartBlock(uint64_t /*block_offset*/) override {}
  // if not use prefix bloom, only call AddKey(key)
  virtual void Add(const Slice& key) override;
  // return num_added_, num of keys
  virtual size_t NumAdded() const override { return num_added_; }
  // only return the slice from LegacyBloomBitsBuilder(format version < 5)
  virtual Slice Finish(const BlockHandle& tmp, Status* status) override;
  using FilterBlockBuilder::Finish;

 protected:
  // call LegacyBloomBitsBuilder(format version < 5), add key and num_added_
  virtual void AddKey(const Slice& key);
  // LegacyBloomBitsBuilder(format version < 5)
  std::unique_ptr<FilterBitsBuilder> filter_bits_builder_;
  // set last_prefix_recorded_ and last_whole_key_recorded_ to false
  virtual void Reset();
  // not used when disable prefix bloom
  void AddPrefix(const Slice& key);
  // return prefix_extractor, however we donot use prefix bloom
  // no need for prefix_extractor
  const SliceTransform* prefix_extractor() { return prefix_extractor_; }

 private:
  // important: all of these might point to invalid addresses
  // at the time of destruction of this filter block. destructor
  // should NOT dereference them.
  const SliceTransform* prefix_extractor_;
  bool whole_key_filtering_;
  bool last_whole_key_recorded_;
  std::string last_whole_key_str_;
  bool last_prefix_recorded_;
  std::string last_prefix_str_;

  uint32_t num_added_;
  std::unique_ptr<const char[]> filter_data_;

};

// A FilterBlockReader is used to parse filter from SST table.
// KeyMayMatch and PrefixMayMatch would trigger filter checking
class FullFilterBlockReader
    : public FilterBlockReaderCommon<ParsedFullFilterBlock> {
 public:
  // set prefix_extractor if needed
  // In our work, dont use prefix_extractor
  #ifdef ART_PLUS
  FullFilterBlockReader(const BlockBasedTable* t,
                        CachableEntry<ParsedFullFilterBlock>&& filter_block,
                        const int hash_id = 0);
  #else
  FullFilterBlockReader(const BlockBasedTable* t,
                        CachableEntry<ParsedFullFilterBlock>&& filter_block);
  #endif
  // call FullFilterBlockReader() to return std::unique_ptr<FilterBlockReader>
  static std::unique_ptr<FilterBlockReader> Create(
      const BlockBasedTable* table, const ReadOptions& ro,
      FilePrefetchBuffer* prefetch_buffer, bool use_cache, bool prefetch,
      bool pin, BlockCacheLookupContext* lookup_context);
  // always return false
  bool IsBlockBased() override { return false; }
  // call MayMatch(key, no_io, get_context, lookup_context)
  bool KeyMayMatch(const Slice& key, const SliceTransform* prefix_extractor,
                   uint64_t block_offset, const bool no_io,
                   const Slice* const const_ikey_ptr, GetContext* get_context,
                   BlockCacheLookupContext* lookup_context) override;
  // not used in our work
  bool PrefixMayMatch(const Slice& prefix,
                      const SliceTransform* prefix_extractor,
                      uint64_t block_offset, const bool no_io,
                      const Slice* const const_ikey_ptr,
                      GetContext* get_context,
                      BlockCacheLookupContext* lookup_context) override;
  // range check, call MayMatch(range, no_io, nullptr, lookup_context);
  void KeysMayMatch(MultiGetRange* range,
                    const SliceTransform* prefix_extractor,
                    uint64_t block_offset, const bool no_io,
                    BlockCacheLookupContext* lookup_context) override;
  // not used in our work
  void PrefixesMayMatch(MultiGetRange* range,
                        const SliceTransform* prefix_extractor,
                        uint64_t block_offset, const bool no_io,
                        BlockCacheLookupContext* lookup_context) override;
  // call ApproximateFilterBlockMemoryUsage(), return Memory Usage
  size_t ApproximateMemoryUsage() const override;
  // when disable prefix bloom, never call this method
  bool RangeMayExist(const Slice* iterate_upper_bound, const Slice& user_key,
                     const SliceTransform* prefix_extractor,
                     const Comparator* comparator,
                     const Slice* const const_ikey_ptr, bool* filter_checked,
                     bool need_upper_bound_check, bool no_io,
                     BlockCacheLookupContext* lookup_context) override;

 private:
  #ifdef ART_PLUS
  // Get From Cache Or Read From SST, to get filter, then check whether entry hit
  bool MayMatch(const Slice& entry, bool no_io, GetContext* get_context,
                BlockCacheLookupContext* lookup_context, const int hash_id = 0) const;
  // range is the key range in the SST, check out these keys may fit in the filter
  void MayMatch(MultiGetRange* range, bool no_io,
                const SliceTransform* prefix_extractor,
                BlockCacheLookupContext* lookup_context, 
                const int hash_id = 0) const;
  #else
  bool MayMatch(const Slice& entry, bool no_io, GetContext* get_context,
                BlockCacheLookupContext* lookup_context) const;
  // range is the key range in the SST, check out these keys may fit in the filter
  void MayMatch(MultiGetRange* range, bool no_io,
                const SliceTransform* prefix_extractor,
                BlockCacheLookupContext* lookup_context) const;
  #endif
  // when disable prefix bloom, never call this method
  bool IsFilterCompatible(const Slice* iterate_upper_bound, const Slice& prefix,
                          const Comparator* comparator) const;

 private:
  bool full_length_enabled_;
  size_t prefix_extractor_full_length_;
  #ifdef ART_PLUS
  const int hash_id_;
  #endif
};

}  // namespace ROCKSDB_NAMESPACE
