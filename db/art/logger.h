//
// Created by joechen on 22-5-11.
// Copy from MatrixKV
//

#pragma once
#include <string>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

#define RECORD_INFO(format, ...) WriteLog(format, ##__VA_ARGS__)

#define RECORD_DEBUG(format, ...) WriteDebug(format, ##__VA_ARGS__)

void SetLogPath(const std::string& path);

extern void WriteLog(const char* format, ...);

extern void WriteDebug(const char* format, ...);

inline uint64_t GetNowMicros() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec) * 1000000 + tv.tv_usec;
}

inline uint64_t GetStartTime() {
  static uint64_t start_time = GetNowMicros();
  return GetNowMicros() - start_time;
}

}  // namespace ROCKSDB_NAMESPACE