//
// Created by joechen on 22-5-11.
//

#include "logger.h"

#include <mutex>

namespace ROCKSDB_NAMESPACE {

const std::string compaction_log = "/tmp/compaction_art.csv";

const std::string debug_log = "/tmp/debug_art.txt";

void InitLogFile() {
  FILE* fp = fopen(compaction_log.c_str(), "a");
  if (fp == nullptr) {
    printf("log failed\n");
  }
  fclose(fp);
  RECORD_INFO(
      "id, read(MB), write(MB), read_write_amp, write_amp"
      ", time(s), start(s), is_level0\n");
}

std::mutex m;

void WriteLog(const char* format, ...) {
  std::lock_guard<std::mutex> write_lk(m);

  va_list ap;
  va_start(ap, format);
  char buf[8192];
  vsprintf(buf, format, ap);
  va_end(ap);

  FILE* fp = fopen(compaction_log.c_str(), "a");
  if (fp == nullptr) {
    printf("log failed\n");
  }

  fprintf(fp, "%s", buf);
  fclose(fp);
}

void WriteDebug(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  char buf[8192];
  vsprintf(buf, format, ap);
  va_end(ap);

  FILE* fp = fopen(debug_log.c_str(), "a");
  if (fp == nullptr) {
    printf("log failed\n");
  }

  fprintf(fp, "%s", buf);
  fclose(fp);
}

}  // namespace ROCKSDB_NAMESPACE