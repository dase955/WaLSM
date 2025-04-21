//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#include "table/block_based/block_based_table_segment_aware_iterator.h"
#include <cstdint>
#include <memory>
#include "db/dbformat.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {
bool BlockBasedTableSegmentAwareIterator::Valid() const {
  return data_iter_ && data_iter_->Valid();
}

void BlockBasedTableSegmentAwareIterator::SeekToFirst() {
  data_iter_->SeekToFirst();
  SeekFilterAndUpdateSegmentID();
}

void BlockBasedTableSegmentAwareIterator::SeekToLast() {
  data_iter_->SeekToLast();
  SeekFilterAndUpdateSegmentID();
}

void BlockBasedTableSegmentAwareIterator::Seek(const Slice& target) {
  // data_iter receives internal key, while filter_index_iter receives modified_key (maybe internal key or just internal key)
  data_iter_->Seek(target);
  SeekFilterAndUpdateSegmentID();
}

void BlockBasedTableSegmentAwareIterator::SeekForPrev(const Slice& target) {
  data_iter_->SeekForPrev(target);
  SeekFilterAndUpdateSegmentID();
}

void BlockBasedTableSegmentAwareIterator::Next() {
  data_iter_->Next();
  UpdateSegmentID();
}

void BlockBasedTableSegmentAwareIterator::Prev() {
  data_iter_->Prev();
  // degraded performance
  SeekFilterAndUpdateSegmentID();
}

Slice BlockBasedTableSegmentAwareIterator::key() const {
  return data_iter_->key();
}

Slice BlockBasedTableSegmentAwareIterator::user_key() const {
  return data_iter_->user_key();
}

Slice BlockBasedTableSegmentAwareIterator::value() const {
  return data_iter_->value();
}

Status BlockBasedTableSegmentAwareIterator::status() const {
  Status data_iter_status = data_iter_->status();
  if (!data_iter_status.ok()) {
    return data_iter_status;
  }
  return status_;
}

uint32_t BlockBasedTableSegmentAwareIterator::segment_id() const {
  return current_segment_id_;
}

void BlockBasedTableSegmentAwareIterator::SeekFilterAndUpdateSegmentID() {
  if (!data_iter_->Valid()) {
    status_ = data_iter_->status();
    current_segment_id_ = INVALID_SEGMENT_ID;
    return;
  }

  // we should return segment_id corresponding to the user_key() when called segment_id(),
  // so here we use data_iter_->user_key() to get the user_key for filter_index_iter
  Slice current_user_key = data_iter_->user_key();

  std::unique_ptr<const char[]> current_modified_key_buf;
  Slice current_modified_key = generate_modified_user_key(
      current_modified_key_buf, current_user_key, 0, 0);

  filter_index_iter_->Seek(current_modified_key);
  UpdateSegmentID();
}

// assumes we will get the entire filter partition key (including user_key, seq_num, segment_id)
// may iterate over the filter_index_iter to find correct filter_index, then extract segment_id
void BlockBasedTableSegmentAwareIterator::UpdateSegmentID() {
  if (!data_iter_->Valid()) {
    current_segment_id_ = INVALID_SEGMENT_ID;
    return;
  }

  if (!filter_index_iter_ || !filter_index_iter_->Valid()) {
    current_segment_id_ = INVALID_SEGMENT_ID;
    return;
  }

  Slice current_user_key = data_iter_->user_key();

  std::unique_ptr<const char[]> current_modified_key_buf;
  Slice current_modified_key = generate_modified_user_key(
      current_modified_key_buf, current_user_key, 0, 0);

  Slice filter_key = filter_index_iter_->user_key();
  // forward lookup
  while (segment_id_removing_comparator_->Compare(current_modified_key, filter_key) > 0) {
    filter_index_iter_->Next();
    if (!filter_index_iter_->Valid()) {
      current_segment_id_ = INVALID_SEGMENT_ID;
      return;
    }
    filter_key = filter_index_iter_->user_key();
  }
  // backward lookup not implemented
  // do nothing here, since we already seek it
  // frequently seeking backward is not good for performance


  uint32_t filter_index = DecodeFixed32R(filter_key.data());
  if (filter_index > 0) {
    // filter_index=0 should always be satisfied
    current_segment_id_ = INVALID_SEGMENT_ID;
    return;
  }

  uint32_t segment_id = INVALID_SEGMENT_ID;
  if (filter_key.size() >= 8) {
    segment_id = DecodeFixed32R(filter_key.data() + filter_key.size() - 4);
  }

  current_segment_id_ = segment_id;
}

}  // namespace ROCKSDB_NAMESPACE
