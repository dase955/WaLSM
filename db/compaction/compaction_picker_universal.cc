//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/compaction/compaction_picker_universal.h"
#ifndef ROCKSDB_LITE

#include <cinttypes>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include "db/column_family.h"
#include "file/filename.h"
#include "logging/log_buffer.h"
#include "monitoring/statistics.h"
#include "test_util/sync_point.h"
#include "util/random.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {
namespace {
// A helper class that form universal compactions. The class is used by
// UniversalCompactionPicker::PickCompaction().
// The usage is to create the class, and get the compaction object by calling
// PickCompaction().
class UniversalCompactionBuilder {
 public:
  UniversalCompactionBuilder(
      const ImmutableCFOptions& ioptions, const InternalKeyComparator* icmp,
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options, VersionStorageInfo* vstorage,
      UniversalCompactionPicker* picker, LogBuffer* log_buffer)
      : ioptions_(ioptions),
        icmp_(icmp),
        cf_name_(cf_name),
        mutable_cf_options_(mutable_cf_options),
        mutable_db_options_(mutable_db_options),
        vstorage_(vstorage),
        picker_(picker),
        log_buffer_(log_buffer) {}

  // Form and return the compaction object. The caller owns return object.
  Compaction* PickCompaction();

 private:
  struct SortedRun {
    SortedRun(int _level, FileMetaData* _file, uint64_t _size,
              uint64_t _compensated_file_size, bool _being_compacted)
        : level(_level),
          file(_file),
          size(_size),
          compensated_file_size(_compensated_file_size),
          being_compacted(_being_compacted) {
      assert(compensated_file_size > 0);
      assert(level != 0 || file != nullptr);
    }

    void Dump(char* out_buf, size_t out_buf_size,
              bool print_path = false) const;

    // sorted_run_count is added into the string to print
    void DumpSizeInfo(char* out_buf, size_t out_buf_size,
                      size_t sorted_run_count) const;

    int level;
    // `file` Will be null for level > 0. For level = 0, the sorted run is
    // for this file.
    FileMetaData* file;
    // For level > 0, `size` and `compensated_file_size` are sum of sizes all
    // files in the level. `being_compacted` should be the same for all files
    // in a non-zero level. Use the value here.
    uint64_t size;
    uint64_t compensated_file_size;
    bool being_compacted;
  };

  Compaction* PickCompactionForL0();

  Compaction* PickCompactionForSizeMarked();

  // Used in universal compaction when the enabled_trivial_move
  // option is set. Checks whether there are any overlapping files
  // in the input. Returns true if the input files are non
  // overlapping.
  bool IsInputFilesNonOverlapping(Compaction* c);

  const ImmutableCFOptions& ioptions_;
  const InternalKeyComparator* icmp_;
  double score_;
  std::vector<SortedRun> sorted_runs_;
  const std::string& cf_name_;
  const MutableCFOptions& mutable_cf_options_;
  const MutableDBOptions& mutable_db_options_;
  VersionStorageInfo* vstorage_;
  UniversalCompactionPicker* picker_;
  LogBuffer* log_buffer_;

  static std::vector<SortedRun> CalculateSortedRuns(
      const VersionStorageInfo& vstorage);

  static std::vector<SortedRun> CalculateSortedRuns(
      const VersionStorageInfo::FilePartition& file_partition);

  // Pick a path ID to place a newly generated file, with its estimated file
  // size.
  static uint32_t GetPathId(const ImmutableCFOptions& ioptions,
                            const MutableCFOptions& mutable_cf_options,
                            uint64_t file_size);
};

// Used in universal compaction when trivial move is enabled.
// This structure is used for the construction of min heap
// that contains the file meta data, the level of the file
// and the index of the file in that level

struct InputFileInfo {
  InputFileInfo() : f(nullptr), level(0), index(0) {}

  FileMetaData* f;
  size_t level;
  size_t index;
};

// Used in universal compaction when trivial move is enabled.
// This comparator is used for the construction of min heap
// based on the smallest key of the file.
struct SmallestKeyHeapComparator {
  explicit SmallestKeyHeapComparator(const Comparator* ucmp) { ucmp_ = ucmp; }

  bool operator()(InputFileInfo i1, InputFileInfo i2) const {
    return (ucmp_->Compare(i1.f->smallest.user_key(),
                           i2.f->smallest.user_key()) > 0);
  }

 private:
  const Comparator* ucmp_;
};

typedef std::priority_queue<InputFileInfo, std::vector<InputFileInfo>,
                            SmallestKeyHeapComparator>
    SmallestKeyHeap;

// This function creates the heap that is used to find if the files are
// overlapping during universal compaction when the allow_trivial_move
// is set.
SmallestKeyHeap create_level_heap(Compaction* c, const Comparator* ucmp) {
  SmallestKeyHeap smallest_key_priority_q =
      SmallestKeyHeap(SmallestKeyHeapComparator(ucmp));

  InputFileInfo input_file;

  for (size_t l = 0; l < c->num_input_levels(); l++) {
    if (c->num_input_files(l) != 0) {
      if (l == 0 && c->start_level() == 0) {
        for (size_t i = 0; i < c->num_input_files(0); i++) {
          input_file.f = c->input(0, i);
          input_file.level = 0;
          input_file.index = i;
          smallest_key_priority_q.push(std::move(input_file));
        }
      } else {
        input_file.f = c->input(l, 0);
        input_file.level = l;
        input_file.index = 0;
        smallest_key_priority_q.push(std::move(input_file));
      }
    }
  }
  return smallest_key_priority_q;
}

#ifndef NDEBUG
// smallest_seqno and largest_seqno are set iff. `files` is not empty.
void GetSmallestLargestSeqno(const std::vector<FileMetaData*>& files,
                             SequenceNumber* smallest_seqno,
                             SequenceNumber* largest_seqno) {
  bool is_first = true;
  for (FileMetaData* f : files) {
    assert(f->fd.smallest_seqno <= f->fd.largest_seqno);
    if (is_first) {
      is_first = false;
      *smallest_seqno = f->fd.smallest_seqno;
      *largest_seqno = f->fd.largest_seqno;
    } else {
      if (f->fd.smallest_seqno < *smallest_seqno) {
        *smallest_seqno = f->fd.smallest_seqno;
      }
      if (f->fd.largest_seqno > *largest_seqno) {
        *largest_seqno = f->fd.largest_seqno;
      }
    }
  }
}
#endif
}  // namespace

// Algorithm that checks to see if there are any overlapping
// files in the input
bool UniversalCompactionBuilder::IsInputFilesNonOverlapping(Compaction* c) {
  auto comparator = icmp_->user_comparator();
  int first_iter = 1;

  InputFileInfo prev, curr, next;

  SmallestKeyHeap smallest_key_priority_q =
      create_level_heap(c, icmp_->user_comparator());

  while (!smallest_key_priority_q.empty()) {
    curr = smallest_key_priority_q.top();
    smallest_key_priority_q.pop();

    if (first_iter) {
      prev = curr;
      first_iter = 0;
    } else {
      if (comparator->Compare(prev.f->largest.user_key(),
                              curr.f->smallest.user_key()) >= 0) {
        // found overlapping files, return false
        return false;
      }
      assert(comparator->Compare(curr.f->largest.user_key(),
                                 prev.f->largest.user_key()) > 0);
      prev = curr;
    }

    next.f = nullptr;

    if (c->level(curr.level) != 0 &&
        curr.index < c->num_input_files(curr.level) - 1) {
      next.f = c->input(curr.level, curr.index + 1);
      next.level = curr.level;
      next.index = curr.index + 1;
    }

    if (next.f) {
      smallest_key_priority_q.push(std::move(next));
    }
  }
  return true;
}

bool UniversalCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  if (vstorage->GetL0CompactionScore() >= 1.0) {
    return true;
  }
  if (!vstorage->FilesMarkedForCompaction().empty()) {
    return true;
  }
  return false;
}

bool UniversalCompactionPicker::NeedsMerge(
    const VersionStorageInfo*) const {
  return true;
}

Compaction* UniversalCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, SequenceNumber /* earliest_memtable_seqno */) {
  UniversalCompactionBuilder builder(ioptions_, icmp_, cf_name,
                                     mutable_cf_options, mutable_db_options,
                                     vstorage, this, log_buffer);
  return builder.PickCompaction();
}

void UniversalCompactionBuilder::SortedRun::Dump(char* out_buf,
                                                 size_t out_buf_size,
                                                 bool print_path) const {
  if (level == 0) {
    assert(file != nullptr);
    if (file->fd.GetPathId() == 0 || !print_path) {
      snprintf(out_buf, out_buf_size, "file %" PRIu64, file->fd.GetNumber());
    } else {
      snprintf(out_buf, out_buf_size, "file %" PRIu64
                                      "(path "
                                      "%" PRIu32 ")",
               file->fd.GetNumber(), file->fd.GetPathId());
    }
  } else {
    snprintf(out_buf, out_buf_size, "level %d", level);
  }
}

void UniversalCompactionBuilder::SortedRun::DumpSizeInfo(
    char* out_buf, size_t out_buf_size, size_t sorted_run_count) const {
  if (level == 0) {
    assert(file != nullptr);
    snprintf(out_buf, out_buf_size,
             "file %" PRIu64 "[%" ROCKSDB_PRIszt
             "] "
             "with size %" PRIu64 " (compensated size %" PRIu64 ")",
             file->fd.GetNumber(), sorted_run_count, file->fd.GetFileSize(),
             file->compensated_file_size);
  } else {
    snprintf(out_buf, out_buf_size,
             "level %d[%" ROCKSDB_PRIszt
             "] "
             "with size %" PRIu64 " (compensated size %" PRIu64 ")",
             level, sorted_run_count, size, compensated_file_size);
  }
}

std::vector<UniversalCompactionBuilder::SortedRun>
UniversalCompactionBuilder::CalculateSortedRuns(
    const VersionStorageInfo& vstorage) {
  std::vector<UniversalCompactionBuilder::SortedRun> ret;
  for (FileMetaData* f : vstorage.LevelFiles(0)) {
    ret.emplace_back(0, f, f->fd.GetFileSize(), f->compensated_file_size,
                     f->being_compacted);
  }
  for (int level = 1; level < vstorage.num_levels(); level++) {
    uint64_t total_compensated_size = 0U;
    uint64_t total_size = 0U;
    bool being_compacted = false;
    for (FileMetaData* f : vstorage.LevelFiles(level)) {
      total_compensated_size += f->compensated_file_size;
      total_size += f->fd.GetFileSize();
      // Size amp, read amp and periodic compactions always include all files
      // for a non-zero level. However, a delete triggered compaction and
      // a trivial move might pick a subset of files in a sorted run. So
      // always check all files in a sorted run and mark the entire run as
      // being compacted if one or more files are being compacted
      if (f->being_compacted) {
        being_compacted = f->being_compacted;
      }
    }
    if (total_compensated_size > 0) {
      ret.emplace_back(level, nullptr, total_size, total_compensated_size,
                       being_compacted);
    }
  }
  return ret;
}

std::vector<UniversalCompactionBuilder::SortedRun>
UniversalCompactionBuilder::CalculateSortedRuns(
    const VersionStorageInfo::FilePartition& file_partition) {
  std::vector<UniversalCompactionBuilder::SortedRun> ret;
  for (int level = 1; level < file_partition.level_; level++) {
    uint64_t total_compensated_size = 0U;
    uint64_t total_size = 0U;
    bool being_compacted = false;
    for (FileMetaData* f : file_partition.files_[level]) {
      total_compensated_size += f->compensated_file_size;
      total_size += f->fd.GetFileSize();
      // Size amp, read amp and periodic compactions always include all files
      // for a non-zero level. However, a delete triggered compaction and
      // a trivial move might pick a subset of files in a sorted run. So
      // always check all files in a sorted run and mark the entire run as
      // being compacted if one or more files are being compacted
      if (f->being_compacted) {
        being_compacted = f->being_compacted;
      }
    }
    if (total_compensated_size > 0) {
      ret.emplace_back(level, nullptr, total_size, total_compensated_size,
                       being_compacted);
    }
  }
  return ret;
}

// Universal style of compaction. Pick files that are contiguous in
// time-range to compact.
Compaction* UniversalCompactionBuilder::PickCompaction() {
  score_ = vstorage_->GetL0CompactionScore();
  sorted_runs_ = CalculateSortedRuns(*vstorage_);

  if (sorted_runs_.empty() ||
      (vstorage_->FilesMarkedForCompaction().empty() &&
       sorted_runs_.size() < (unsigned int)mutable_cf_options_
                                 .level0_file_num_compaction_trigger)) {
    ROCKS_LOG_BUFFER(log_buffer_, "[%s] Universal: nothing to do\n",
                     cf_name_.c_str());
    TEST_SYNC_POINT_CALLBACK(
        "UniversalCompactionBuilder::PickCompaction:Return", nullptr);
    return nullptr;
  }

  VersionStorageInfo::LevelSummaryStorage tmp;
  ROCKS_LOG_BUFFER_MAX_SZ(
      log_buffer_, 3072,
      "[%s] Universal: sorted runs: %" ROCKSDB_PRIszt " files: %s\n",
      cf_name_.c_str(), sorted_runs_.size(), vstorage_->LevelSummary(&tmp));

  Compaction* c = nullptr;

  if (sorted_runs_.size() >=
      static_cast<size_t>(mutable_cf_options_.level0_file_num_compaction_trigger)) {
    c = PickCompactionForL0();
  }

  if (c == nullptr) {
    c = PickCompactionForSizeMarked();
  }

  if (mutable_cf_options_.compaction_options_universal.allow_trivial_move ==
          true &&
      c->compaction_reason() != CompactionReason::kPeriodicCompaction) {
    c->set_is_trivial_move(IsInputFilesNonOverlapping(c));
  }

// validate that all the chosen files of L0 are non overlapping in time
#ifndef NDEBUG
  bool is_first = true;

  size_t level_index = 0U;
  if (c->start_level() == 0) {
    for (auto f : *c->inputs(0)) {
      assert(f->fd.smallest_seqno <= f->fd.largest_seqno);
      if (is_first) {
        is_first = false;
      }
    }
    level_index = 1U;
  }
  for (; level_index < c->num_input_levels(); level_index++) {
    if (c->num_input_files(level_index) != 0) {
      SequenceNumber smallest_seqno = 0U;
      SequenceNumber largest_seqno = 0U;
      GetSmallestLargestSeqno(*(c->inputs(level_index)), &smallest_seqno,
                              &largest_seqno);
      if (is_first) {
        is_first = false;
      }
    }
  }
#endif
  // update statistics
  RecordInHistogram(ioptions_.statistics, NUM_FILES_IN_SINGLE_COMPACTION,
                    c->inputs(0)->size());

  picker_->RegisterCompaction(c);
  vstorage_->ComputeCompactionScore(ioptions_, mutable_cf_options_);

  TEST_SYNC_POINT_CALLBACK("UniversalCompactionBuilder::PickCompaction:Return",
                           c);
  return c;
}

uint32_t UniversalCompactionBuilder::GetPathId(
    const ImmutableCFOptions& ioptions,
    const MutableCFOptions& mutable_cf_options, uint64_t file_size) {
  // Two conditions need to be satisfied:
  // (1) the target path needs to be able to hold the file's size
  // (2) Total size left in this and previous paths need to be not
  //     smaller than expected future file size before this new file is
  //     compacted, which is estimated based on size_ratio.
  // For example, if now we are compacting files of size (1, 1, 2, 4, 8),
  // we will make sure the target file, probably with size of 16, will be
  // placed in a path so that eventually when new files are generated and
  // compacted to (1, 1, 2, 4, 8, 16), all those files can be stored in or
  // before the path we chose.
  //
  // TODO(sdong): now the case of multiple column families is not
  // considered in this algorithm. So the target size can be violated in
  // that case. We need to improve it.
  uint64_t accumulated_size = 0;
  uint64_t future_size =
      file_size *
      (100 - mutable_cf_options.compaction_options_universal.size_ratio) / 100;
  uint32_t p = 0;
  assert(!ioptions.cf_paths.empty());
  for (; p < ioptions.cf_paths.size() - 1; p++) {
    uint64_t target_size = ioptions.cf_paths[p].target_size;
    if (target_size > file_size &&
        accumulated_size + (target_size - file_size) > future_size) {
      return p;
    }
    accumulated_size += target_size;
  }
  return p;
}


// For L0 sorted runs number compaction trigger
Compaction* UniversalCompactionBuilder::PickCompactionForL0() {
  assert(!sorted_runs_.empty());

  int num_sr_not_compacted = 0;
  for (size_t i = 0; i < sorted_runs_.size(); i++) {
    if (sorted_runs_[i].level == 0 && !sorted_runs_[i].being_compacted) {
      num_sr_not_compacted++;
    }
  }

  if (num_sr_not_compacted >=
      mutable_cf_options_.level0_file_num_compaction_trigger) {
    uint64_t estimated_total_size = 0;
    const int output_level = vstorage_->num_levels() - 1,
              start_level = 0;

    // initialize inputs levels
    std::vector<CompactionInputFiles> inputs(vstorage_->num_levels());
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputs[i].level = start_level + static_cast<int>(i);
    }

    for (size_t i = 0; i < sorted_runs_.size(); i++) {
      auto& picking_sr = sorted_runs_[i];
      FileMetaData* picking_file = picking_sr.file;
      if (sorted_runs_[i].level == 0
          && !sorted_runs_[i].being_compacted) {
        estimated_total_size += sorted_runs_[i].size;
        inputs[0].files.push_back(picking_file);
      }
      char file_num_buf[256];
      picking_sr.DumpSizeInfo(file_num_buf, sizeof(file_num_buf), i);
      ROCKS_LOG_BUFFER(log_buffer_, "[%s] Universal: Picking %s",
                       cf_name_.c_str(), file_num_buf);
    }

    uint32_t path_id =
        GetPathId(ioptions_, mutable_cf_options_, estimated_total_size);
    return new Compaction(
        vstorage_, ioptions_, mutable_cf_options_, mutable_db_options_,
        std::move(inputs), output_level,
        MaxFileSizeForLevel(mutable_cf_options_, output_level,
                            kCompactionStyleUniversal),
        LLONG_MAX, path_id,
        GetCompressionType(ioptions_, vstorage_, mutable_cf_options_,
                           output_level, 1, true /* enable_compression */),
        GetCompressionOptions(mutable_cf_options_, vstorage_, output_level,
                              true /* enable_compression */),
        /* max_subcompactions */ 0, /* grandparents */ {}, /* is manual */ false,
        score_, false /* deletion_compaction */, CompactionReason::kUniversalSortedRunNum
        );
  }

  return nullptr;
}

Compaction* UniversalCompactionBuilder::
    PickCompactionForSizeMarked() {
  auto& files = vstorage_->FilesMarkedForCompaction();
  if (files.empty()) {
    return nullptr;
  }

  for (auto& p : files) {
    if (p.second->being_compacted) return nullptr;
  }

  const int input_level = files[0].first;
  const int output_level = input_level - 1;
  uint64_t estimated_total_size = 0;
  std::vector<CompactionInputFiles> inputs(vstorage_->num_levels());
  for (int i = 0; i < vstorage_->num_levels(); i++) {
    inputs[i].level = i;
  }
  for (unsigned int i = 0; i < files.size(); i++) {
    inputs[input_level].files.push_back(files[i].second);
    estimated_total_size += files[i].second->fd.GetFileSize();
  }

  uint32_t path_id =
      GetPathId(ioptions_, mutable_cf_options_, estimated_total_size);
  return new Compaction(
      vstorage_, ioptions_, mutable_cf_options_, mutable_db_options_,
      std::move(inputs), output_level,
      LLONG_MAX,
      LLONG_MAX, path_id,
      GetCompressionType(ioptions_, vstorage_, mutable_cf_options_,
                         output_level, 1, true /* enable_compression */),
      GetCompressionOptions(mutable_cf_options_, vstorage_, output_level,
                            true /* enable_compression */),
      /* max_subcompactions */ 0, /* grandparents */ {}, /* is manual */ false,
      score_, false /* deletion_compaction */, CompactionReason::kUniversalSortedRunNum
  );
}

}  // namespace ROCKSDB_NAMESPACE

#endif  // !ROCKSDB_LITE
