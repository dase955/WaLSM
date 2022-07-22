//
// Created by joechen on 22-5-11.
//

#include "logger.h"

#include <unistd.h>
#include <mutex>

namespace ROCKSDB_NAMESPACE {

std::string log_path = "/tmp";

std::string compaction_filename = "compaction_art.txt";

std::string compaction_log = "/tmp/compaction_art.txt";

std::string debug_filename = "debug_art.txt";

std::string debug_log = "/tmp/debug_art.txt";

void InitLogFile() {
  FILE* fp;

  fp = fopen(compaction_log.c_str(), "w");
  if (fp == nullptr) {
    printf("log failed\n");
  }
  fclose(fp);

  fp = fopen(debug_log.c_str(), "w");
  if (fp == nullptr) {
    printf("log failed\n");
  }
  fclose(fp);
}

void SetLogPath(const std::string& path) {
  log_path = path;
  compaction_log = log_path + "/" + compaction_filename;
  debug_log = log_path + "/" + debug_filename;
  InitLogFile();
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
  std::lock_guard<std::mutex> write_lk(m);

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