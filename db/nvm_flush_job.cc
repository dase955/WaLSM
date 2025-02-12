//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/nvm_flush_job.h"

#include <cinttypes>

#include <algorithm>
#include <vector>

#include "db/art/logger.h"
#include "db/builder.h"
#include "db/db_iter.h"
#include "db/dbformat.h"
#include "db/event_helpers.h"
#include "db/log_reader.h"
#include "db/log_writer.h"
#include "db/memtable.h"
#include "db/memtable_list.h"
#include "db/merge_context.h"
#include "db/range_tombstone_fragmenter.h"
#include "db/version_set.h"
#include "file/file_util.h"
#include "file/filename.h"
#include "logging/event_logger.h"
#include "logging/log_buffer.h"
#include "logging/logging.h"
#include "monitoring/iostats_context_imp.h"
#include "monitoring/perf_context_imp.h"
#include "monitoring/thread_status_util.h"
#include "port/port.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"
#include "rocksdb/table.h"
#include "table/merging_iterator.h"
#include "table/table_builder.h"
#include "table/two_level_iterator.h"
#include "test_util/sync_point.h"
#include "util/coding.h"
#include "util/mutexlock.h"
#include "util/stop_watch.h"

namespace ROCKSDB_NAMESPACE {

// WaLSM+ Note: copy of FlushJob, add nvm reading
NVMFlushJob::NVMFlushJob(SingleCompactionJob* job,
    const std::string& dbname, ColumnFamilyData* cfd,
    const ImmutableDBOptions& db_options,
    const MutableCFOptions& mutable_cf_options,
    const FileOptions& file_options, VersionSet* versions,
    InstrumentedMutex* db_mutex, std::atomic<bool>* shutting_down,
    JobContext* job_context,
    LogBuffer* log_buffer, FSDirectory* db_directory,
    FSDirectory* output_file_directory, CompressionType output_compression,
    Statistics* stats, EventLogger* event_logger, bool measure_io_stats,
    const bool sync_output_directory, const bool write_manifest,
    const std::shared_ptr<IOTracer>& io_tracer,
    const std::string& db_id, const std::string& db_session_id)
    : dbname_(dbname),
      db_id_(db_id),
      db_session_id_(db_session_id),
      cfd_(cfd),
      db_options_(db_options),
      mutable_cf_options_(mutable_cf_options),
      file_options_(file_options),
      versions_(versions),
      db_mutex_(db_mutex),
      shutting_down_(shutting_down),
      job_context_(job_context),
      log_buffer_(log_buffer),
      db_directory_(db_directory),
      output_file_directory_(output_file_directory),
      output_compression_(output_compression),
      stats_(stats),
      event_logger_(event_logger),
      measure_io_stats_(measure_io_stats),
      sync_output_directory_(sync_output_directory),
      write_manifest_(write_manifest),
      edit_(nullptr),
      base_(nullptr),
      job_(job),
      io_tracer_(io_tracer) {
  // Update the thread status to indicate flush.
  ReportStartedFlush();
  TEST_SYNC_POINT("FlushJob::FlushJob()");
}

NVMFlushJob::~NVMFlushJob() {
  io_status_.PermitUncheckedError();
  ThreadStatusUtil::ResetThreadStatus();
}

void NVMFlushJob::ReportStartedFlush() {
  ThreadStatusUtil::SetColumnFamily(cfd_, cfd_->ioptions()->env,
                                    db_options_.enable_thread_tracking);
  ThreadStatusUtil::SetThreadOperation(ThreadStatus::OP_FLUSH);
  ThreadStatusUtil::SetThreadOperationProperty(
      ThreadStatus::COMPACTION_JOB_ID,
      job_context_->job_id);
  IOSTATS_RESET(bytes_written);
}

void NVMFlushJob::RecordFlushIOStats() {
  RecordTick(stats_, FLUSH_WRITE_BYTES, IOSTATS(bytes_written));
  ThreadStatusUtil::IncreaseThreadOperationProperty(
      ThreadStatus::FLUSH_BYTES_WRITTEN, IOSTATS(bytes_written));
  IOSTATS_RESET(bytes_written);
}

void NVMFlushJob::Preprocess() {
  edit_ = cfd_->mem()->GetEdits();
  edit_->SetPrevLogNumber(0);
  // SetLogNumber(log_num) indicates logs with number smaller than log_num
  // will no longer be picked up for recovery.
  edit_->SetLogNumber(cfd_->mem()->GetNextLogNumber());
  edit_->SetColumnFamily(cfd_->GetID());

  // path 0 for level 0 file.
  meta_.fd = FileDescriptor(versions_->NewFileNumber(), 0, 0);

  base_ = cfd_->current();
  base_->Ref();  // it is likely that we do not need this reference

  if (measure_io_stats_) {
    SetPerfLevel(PerfLevel::kEnableTime);
    prev_write_nanos = IOSTATS(write_nanos);
    prev_fsync_nanos = IOSTATS(fsync_nanos);
    prev_range_sync_nanos = IOSTATS(range_sync_nanos);
    prev_prepare_write_nanos = IOSTATS(prepare_write_nanos);
    prev_cpu_write_nanos = IOSTATS(cpu_write_nanos);
    prev_cpu_read_nanos = IOSTATS(cpu_read_nanos);
  }

  start_micros = db_options_.env->NowMicros();
  start_cpu_micros = db_options_.env->NowCPUNanos() / 1000;
  write_hint = cfd_->CalculateSSTWriteHint(0);
  if (log_buffer_) {
    log_buffer_->FlushBufferToLog();
  }
}

// TODO(WaLSM+): we can pass info recorders to BuildTableFromArt()?
void NVMFlushJob::Build() {
  Status s;
  {
    ROCKS_LOG_INFO(db_options_.info_log,
                   "[%s] [JOB %d] Level-0 flush table #%" PRIu64 ": started",
                   cfd_->GetName().c_str(), job_context_->job_id,
                   meta_.fd.GetNumber());

    TEST_SYNC_POINT_CALLBACK("NVMFlushJob::WriteLevel0Table:output_compression",
                             &output_compression_);
    int64_t _current_time = 0;
    auto status = db_options_.env->GetCurrentTime(&_current_time);
    const uint64_t current_time = static_cast<uint64_t>(_current_time);

    uint64_t oldest_key_time = job_->oldest_key_time_;

    // It's not clear whether oldest_key_time is always available. In case
    // it is not available, use current_time.
    uint64_t oldest_ancester_time = std::min(current_time, oldest_key_time);

    TEST_SYNC_POINT_CALLBACK(
        "NVMFlushJob::WriteLevel0Table:oldest_ancester_time",
        &oldest_ancester_time);
    meta_.oldest_ancester_time = oldest_ancester_time;

    meta_.file_creation_time = current_time;

    uint64_t creation_time = meta_.oldest_ancester_time;

    IOStatus io_s;
    s = BuildTableFromArt( // TODO(WaLSM+): pass temp recorders ptrs, update these recorders when building
        job_, dbname_, db_options_.env, db_options_.fs.get(),
        *cfd_->ioptions(), mutable_cf_options_, file_options_,
        cfd_->table_cache(), &meta_,
        cfd_->internal_comparator(),
        cfd_->int_tbl_prop_collector_factories(), cfd_->GetID(),
        cfd_->GetName(),
        // existing_snapshots_,
        // earliest_write_conflict_snapshot_, snapshot_checker_,
        output_compression_,
        mutable_cf_options_.sample_for_compression,
        mutable_cf_options_.compression_opts,
        mutable_cf_options_.paranoid_file_checks, cfd_->internal_stats(),
        TableFileCreationReason::kFlush, &io_s, io_tracer_, event_logger_,
        job_context_->job_id, Env::IO_HIGH, &table_properties_, 0 /* level */,
        creation_time, oldest_key_time, write_hint, current_time, db_id_,
        db_session_id_);
    assert(s.ok());
    if (!io_s.ok()) {
      io_status_ = io_s;
    }
    LogFlush(db_options_.info_log);
  }
  ROCKS_LOG_INFO(db_options_.info_log,
                 "[%s] [JOB %d] Level-0 flush table #%" PRIu64 ": %" PRIu64
                 " bytes %s"
                 "%s",
                 cfd_->GetName().c_str(), job_context_->job_id,
                 meta_.fd.GetNumber(), meta_.fd.GetFileSize(),
                 s.ToString().c_str(),
                 meta_.marked_for_compaction ? " (needs compaction)" : "");

  RECORD_INFO("Flush l0: %.2fMB, %.3lfs, %.3lfs\n",
              meta_.fd.file_size / 1048576.0,
              0.0, GetStartTime() * 1e-6);

  if (s.ok() && output_file_directory_ != nullptr && sync_output_directory_) {
    s = output_file_directory_->Fsync(IOOptions(), nullptr);
  }
}

void NVMFlushJob::PostProcess(InternalStats::CompactionStats& stats) {
  base_->Unref();

  const bool has_output = meta_.fd.GetFileSize() > 0;
  if (has_output) {
    // if we have more than 1 background thread, then we cannot
    // insert files directly into higher levels because some other
    // threads could be concurrently producing compacted files for
    // that key range.
    // Add file to L0
#ifndef EXPERIMENT
    edit_->AddFile(0 /* level */, meta_.fd.GetNumber(), meta_.fd.GetPathId(),
                   meta_.fd.GetFileSize(), meta_.smallest, meta_.largest,
                   meta_.fd.smallest_seqno, meta_.fd.largest_seqno,
                   meta_.marked_for_compaction, meta_.oldest_blob_file_number,
                   meta_.oldest_ancester_time, meta_.file_creation_time,
                   meta_.file_checksum, meta_.file_checksum_func_name);
#endif
  }

  stats.micros = db_options_.env->NowMicros() - start_micros;
  stats.cpu_micros = db_options_.env->NowCPUNanos() / 1000 - start_cpu_micros;
  if (has_output) {
    stats.bytes_written += meta_.fd.GetFileSize();
    stats.num_output_files++;
  }
}

void NVMFlushJob::WriteResult(InternalStats::CompactionStats& stats) {
  cfd_->mem()->SetFlushJobInfo(GetFlushJobInfo());
  RecordTimeToHistogram(stats_, FLUSH_TIME, stats.micros);
  cfd_->internal_stats()->AddCompactionStats(0 /* level */, Env::HIGH, stats);
  cfd_->internal_stats()->AddCFStats(InternalStats::BYTES_FLUSHED,
                                     stats.bytes_written);
  RecordFlushIOStats();

  Status s;
  if (write_manifest_) {
    TEST_SYNC_POINT("NVMFlushJob::InstallResults");
    // Replace immutable memtable with the generated Table
    IOStatus tmp_io_s;
    s = cfd_->imm()->TryInstallNVMFlushResults(
        cfd_, mutable_cf_options_, cfd_->mem(), logs_with_prep_tracker_, versions_, db_mutex_,
        meta_.fd.GetNumber(),
        // &job_context_->memtables_to_free,
        db_directory_, log_buffer_, &committed_flush_jobs_info_, &tmp_io_s);
    if (!tmp_io_s.ok()) {
      io_status_ = tmp_io_s;
    }
  }

  RecordFlushIOStats();

  // When measure_io_stats_ is true, the default 512 bytes is not enough.
  auto stream = event_logger_->LogToBuffer(log_buffer_, 1024);
  stream << "job" << job_context_->job_id << "event"
         << "flush_finished";
  stream << "output_compression"
         << CompressionTypeToString(output_compression_);
  stream << "lsm_state";
  stream.StartArray();
  auto vstorage = cfd_->current()->storage_info();
  for (int level = 0; level < vstorage->num_levels(); ++level) {
    stream << vstorage->NumLevelFiles(level);
  }
  stream.EndArray();

  if (measure_io_stats_) {
    stream << "file_write_nanos" << (IOSTATS(write_nanos) - prev_write_nanos);
    stream << "file_range_sync_nanos"
           << (IOSTATS(range_sync_nanos) - prev_range_sync_nanos);
    stream << "file_fsync_nanos" << (IOSTATS(fsync_nanos) - prev_fsync_nanos);
    stream << "file_prepare_write_nanos"
           << (IOSTATS(prepare_write_nanos) - prev_prepare_write_nanos);
    stream << "file_cpu_write_nanos"
           << (IOSTATS(cpu_write_nanos) - prev_cpu_write_nanos);
    stream << "file_cpu_read_nanos"
           << (IOSTATS(cpu_read_nanos) - prev_cpu_read_nanos);
  }
}

void NVMFlushJob::Cancel() {
  db_mutex_->AssertHeld();
  assert(base_ != nullptr);
  base_->Unref();
}

std::unique_ptr<FlushJobInfo> NVMFlushJob::GetFlushJobInfo() const {
  db_mutex_->AssertHeld();
  std::unique_ptr<FlushJobInfo> info(new FlushJobInfo{});
  info->cf_id = cfd_->GetID();
  info->cf_name = cfd_->GetName();

  const uint64_t file_number = meta_.fd.GetNumber();
  info->file_path =
      MakeTableFileName(cfd_->ioptions()->cf_paths[0].path, file_number);
  info->file_number = file_number;
  info->oldest_blob_file_number = meta_.oldest_blob_file_number;
  info->thread_id = db_options_.env->GetThreadID();
  info->job_id = job_context_->job_id;
  info->smallest_seqno = meta_.fd.smallest_seqno;
  info->largest_seqno = meta_.fd.largest_seqno;
  info->table_properties = table_properties_;
  info->flush_reason = cfd_->GetFlushReason();
  return info;
}

}  // namespace ROCKSDB_NAMESPACE
