//
// Created by joechen on 22-5-11.
//

#include "logger.h"
#include <mutex>

namespace ROCKSDB_NAMESPACE {

const std::string compaction_log = "/tmp/compaction.csv";

void init_log_file() {
  FILE* fp = fopen(compaction_log.c_str(), "w");
  if (fp == nullptr) {
    printf("log failed\n");
  }
  fclose(fp);
  RECORD_INFO("compaction, read(MB), write(MB)"
      ", time(s), start(s), is_level0\n");
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

