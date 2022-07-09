//
// Created by joechen on 22-5-11.
//

#include "logger.h"
#include <mutex>

namespace ROCKSDB_NAMESPACE {

std::string log_path = "/tmp";

std::string compaction_filename = "compaction_nvm_l0.csv";

std::string compaction_log = "/tmp/compaction_nvm_l0.csv";

void SetLogPath(const std::string& path) {
  log_path = path;
  compaction_log = log_path + "/" + compaction_filename;
}

void InitLogFile() {
  FILE* fp = fopen(compaction_log.c_str(), "w");
  if (fp == nullptr) {
    printf("log failed\n");
  }
  fclose(fp);
}

std::mutex m;

void write_log(const char* format, ...) {
  std::lock_guard<std::mutex> lk(m);
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

}  // namespace ROCKSDB_NAMESPACE

