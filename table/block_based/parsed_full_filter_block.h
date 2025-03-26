//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <memory>

#include "table/format.h"

namespace ROCKSDB_NAMESPACE {

class FilterBitsReader;
class FilterPolicy;

// The sharable/cachable part of the full filter.
class ParsedFullFilterBlock {
 public:
  // mainly get FilterBitsReader from filter_policy
  ParsedFullFilterBlock(const FilterPolicy* filter_policy,
                        BlockContents&& contents);
  ~ParsedFullFilterBlock();

  // noticed unique_ptr point to FilterBitsReader, 
  // thus this method return the FilterBitsReader filter_bits_reader_ pointed to
  FilterBitsReader* filter_bits_reader() const {
    return filter_bits_reader_.get();
  }

  // not implemented
  // TODO: consider memory usage of the FilterBitsReader
  size_t ApproximateMemoryUsage() const {
    return block_contents_.ApproximateMemoryUsage();
  }
  // not implemented
  // TODO
  bool own_bytes() const { return block_contents_.own_bytes(); }

 private:
  BlockContents block_contents_;
  std::unique_ptr<FilterBitsReader> filter_bits_reader_;
};

}  // namespace ROCKSDB_NAMESPACE
