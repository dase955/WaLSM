//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once
#include <cstdint>
#include <memory>
#include "db/dbformat.h"
#include "rocksdb/comparator.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_table_reader.h"

#include "table/block_based/block_based_table_reader_impl.h"
#include "table/block_based/block_prefetcher.h"
#include "table/block_based/reader_common.h"
#include "table/internal_iterator.h"

namespace ROCKSDB_NAMESPACE {
// Iterates over the contents of BlockBasedTable, also provides segment_id information by iterating over the filter index.
class BlockBasedTableSegmentAwareIterator : public InternalIteratorBase<Slice> {
 public:
  BlockBasedTableSegmentAwareIterator(
      std::unique_ptr<InternalIterator> data_iter, std::unique_ptr<IndexBlockIter> filter_index_iter,
      const InternalKeyComparator& icomp,
      TableReaderCaller caller)
      : data_iter_(std::move(data_iter)),
        filter_index_iter_(std::move(filter_index_iter)),
        icmp_(&icomp),
        segment_id_removing_comparator_(icomp.user_comparator()),
        lookup_context_(caller),
        user_comparator_(icomp.user_comparator()) {}

  ~BlockBasedTableSegmentAwareIterator() {}

  // Ausuming that `target` is the original user key, not the modified key.
  void Seek(const Slice& target) override;
  // Ausuming that `target` is the original user key, not the modified key.
  void SeekForPrev(const Slice& target) override;
  void SeekToFirst() override;
  void SeekToLast() override;
  void Next() final override;
  bool NextAndGetResult(IterateResult* result) override;
  void Prev() override;
  bool Valid() const override;
  Slice key() const override;
  Slice user_key() const override;
  bool PrepareValue() override;
  Slice value() const override;
  Status status() const override;
  uint32_t segment_id() const override;

 private:
  std::unique_ptr<InternalIterator> data_iter_;
  std::unique_ptr<IndexBlockIter> filter_index_iter_;
  const InternalKeyComparator* icmp_;
  const Comparator* segment_id_removing_comparator_;
  const SliceTransform* prefix_extractor_;
  TableReaderCaller lookup_context_;
  InternalKeyComparator user_comparator_;
  HistogramImpl* file_read_hist_;
  uint32_t current_segment_id_ = 0;
  InternalKey current_partition_key_;
  Status status_;
  int level_;

  void UpdateSegmentID();
  void SeekFilterAndUpdateSegmentID();
};
}  // namespace ROCKSDB_NAMESPACE
