//
// Created by joechen on 22-7-10.
//

#ifndef ROCKSDB_UTILS_H
#define ROCKSDB_UTILS_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <cassert>

#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>

int RemoveDirectory(const char *path) {
  DIR *d = opendir(path);
  size_t path_len = strlen(path);
  int r = -1;

  if (d) {
    struct dirent *p;

    r = 0;
    while (!r && (p=readdir(d))) {
      int r2 = -1;
      char *buf;
      size_t len;

      /* Skip the names "." and ".." as we don't want to recurse on them. */
      if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
        continue;

      len = path_len + strlen(p->d_name) + 2;
      buf = (char*)malloc(len);

      if (buf) {
        struct stat statbuf;

        snprintf(buf, len, "%s/%s", path, p->d_name);
        if (!stat(buf, &statbuf)) {
          if (S_ISDIR(statbuf.st_mode))
            r2 = RemoveDirectory(buf);
          else
            r2 = unlink(buf);
        }
        free(buf);
      }
      r = r2;
    }
    closedir(d);
  }

  if (!r) {
    r = rmdir(path);
  }

  return r;
}

class HistogramBucketMapper {
 public:
  HistogramBucketMapper() {
    // If you change this, you also need to change
    // size of array buckets_ in HistogramImpl
    bucketValues_ = {1, 2};
    valueIndexMap_ = {{1, 0}, {2, 1}};
    double bucket_val = static_cast<double>(bucketValues_.back());
    while ((bucket_val = 1.5 * bucket_val) <= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
      bucketValues_.push_back(static_cast<uint64_t>(bucket_val));
      // Extracts two most significant digits to make histogram buckets more
      // human-readable. E.g., 172 becomes 170.
      uint64_t pow_of_ten = 1;
      while (bucketValues_.back() / 10 > 10) {
        bucketValues_.back() /= 10;
        pow_of_ten *= 10;
      }
      bucketValues_.back() *= pow_of_ten;
      valueIndexMap_[bucketValues_.back()] = bucketValues_.size() - 1;
    }
    maxBucketValue_ = bucketValues_.back();
    minBucketValue_ = bucketValues_.front();
  }

  // converts a value to the bucket index.
  size_t IndexForValue(uint64_t value) const {
    if (value >= maxBucketValue_) {
      return bucketValues_.size() - 1;
    } else if (value >= minBucketValue_) {
      std::map<uint64_t, uint64_t>::const_iterator lowerBound =
          valueIndexMap_.lower_bound(value);
      if (lowerBound != valueIndexMap_.end()) {
        return static_cast<size_t>(lowerBound->second);
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  size_t BucketCount() const {
    return bucketValues_.size();
  }

  uint64_t LastValue() const {
    return maxBucketValue_;
  }

  uint64_t FirstValue() const {
    return minBucketValue_;
  }

  uint64_t BucketLimit(const size_t bucketNumber) const {
    assert(bucketNumber < BucketCount());
    return bucketValues_[bucketNumber];
  }

 private:
  std::vector<uint64_t> bucketValues_;
  uint64_t maxBucketValue_;
  uint64_t minBucketValue_;
  std::map<uint64_t, uint64_t> valueIndexMap_;
};

namespace {
const HistogramBucketMapper bucketMapper;
}

struct HistogramStat {
  HistogramStat()
      : num_buckets_(bucketMapper.BucketCount()) {
    assert(num_buckets_ == sizeof(buckets_) / sizeof(*buckets_));
    Clear();
  }

  ~HistogramStat() {}

  HistogramStat(const HistogramStat&) = delete;
  HistogramStat& operator=(const HistogramStat&) = delete;

  void Clear() {
    min_.store(bucketMapper.LastValue(), std::memory_order_relaxed);
    max_.store(0, std::memory_order_relaxed);
    num_.store(0, std::memory_order_relaxed);
    sum_.store(0, std::memory_order_relaxed);
    sum_squares_.store(0, std::memory_order_relaxed);
    for (unsigned int b = 0; b < num_buckets_; b++) {
      buckets_[b].store(0, std::memory_order_relaxed);
    }
  };

  bool Empty() const { return num() == 0; }

  void Add(uint64_t value) {
    // This function is designed to be lock free, as it's in the critical path
    // of any operation. Each individual value is atomic and the order of updates
    // by concurrent threads is tolerable.
    const size_t index = bucketMapper.IndexForValue(value);
    assert(index < num_buckets_);
    buckets_[index].store(buckets_[index].load(std::memory_order_relaxed) + 1,
                          std::memory_order_relaxed);

    uint64_t old_min = min();
    if (value < old_min) {
      min_.store(value, std::memory_order_relaxed);
    }

    uint64_t old_max = max();
    if (value > old_max) {
      max_.store(value, std::memory_order_relaxed);
    }

    num_.store(num_.load(std::memory_order_relaxed) + 1,
               std::memory_order_relaxed);
    sum_.store(sum_.load(std::memory_order_relaxed) + value,
               std::memory_order_relaxed);
    sum_squares_.store(
        sum_squares_.load(std::memory_order_relaxed) + value * value,
        std::memory_order_relaxed);
  }

  void Merge(HistogramStat& other) {
    // This function needs to be performned with the outer lock acquired
    // However, atomic operation on every member is still need, since Add()
    // requires no lock and value update can still happen concurrently
    uint64_t old_min = min();
    uint64_t other_min = other.min();
    while (other_min < old_min &&
           !min_.compare_exchange_weak(old_min, other_min)) {}

    uint64_t old_max = max();
    uint64_t other_max = other.max();
    while (other_max > old_max &&
           !max_.compare_exchange_weak(old_max, other_max)) {}

    num_.fetch_add(other.num(), std::memory_order_relaxed);
    sum_.fetch_add(other.sum(), std::memory_order_relaxed);
    sum_squares_.fetch_add(other.sum_squares(), std::memory_order_relaxed);
    for (unsigned int b = 0; b < num_buckets_; b++) {
      buckets_[b].fetch_add(other.bucket_at(b), std::memory_order_relaxed);
    }

    other.Clear();
  }

  double Median() const {
    return Percentile(50.0);
  }

  double Percentile(double p) const {
    double threshold = num() * (p / 100.0);
    uint64_t cumulative_sum = 0;
    for (unsigned int b = 0; b < num_buckets_; b++) {
      uint64_t bucket_value = bucket_at(b);
      cumulative_sum += bucket_value;
      if (cumulative_sum >= threshold) {
        // Scale linearly within this bucket
        uint64_t left_point = (b == 0) ? 0 : bucketMapper.BucketLimit(b-1);
        uint64_t right_point = bucketMapper.BucketLimit(b);
        uint64_t left_sum = cumulative_sum - bucket_value;
        uint64_t right_sum = cumulative_sum;
        double pos = 0;
        uint64_t right_left_diff = right_sum - left_sum;
        if (right_left_diff != 0) {
          pos = (threshold - left_sum) / right_left_diff;
        }
        double r = left_point + (right_point - left_point) * pos;
        uint64_t cur_min = min();
        uint64_t cur_max = max();
        if (r < cur_min) r = static_cast<double>(cur_min);
        if (r > cur_max) r = static_cast<double>(cur_max);
        return r;
      }
    }
    return static_cast<double>(max());
  }

  double Average() const {
    uint64_t cur_num = num();
    uint64_t cur_sum = sum();
    if (cur_num == 0) return 0;
    return static_cast<double>(cur_sum) / static_cast<double>(cur_num);
  }

  double StandardDeviation() const {
    uint64_t cur_num = num();
    uint64_t cur_sum = sum();
    uint64_t cur_sum_squares = sum_squares();
    if (cur_num == 0) return 0;
    double variance =
        static_cast<double>(cur_sum_squares * cur_num - cur_sum * cur_sum) /
        static_cast<double>(cur_num * cur_num);
    return std::sqrt(variance);
  }

  inline uint64_t min() const { return min_.load(std::memory_order_relaxed); }
  inline uint64_t max() const { return max_.load(std::memory_order_relaxed); }
  inline uint64_t num() const { return num_.load(std::memory_order_relaxed); }
  inline uint64_t sum() const { return sum_.load(std::memory_order_relaxed); }
  inline uint64_t sum_squares() const {
    return sum_squares_.load(std::memory_order_relaxed);
  }
  inline uint64_t bucket_at(size_t b) const {
    return buckets_[b].load(std::memory_order_relaxed);
  }

  // To be able to use HistogramStat as thread local variable, it
  // cannot have dynamic allocated member. That's why we're
  // using manually values from BucketMapper
  std::atomic_uint_fast64_t min_;
  std::atomic_uint_fast64_t max_;
  std::atomic_uint_fast64_t num_;
  std::atomic_uint_fast64_t sum_;
  std::atomic_uint_fast64_t sum_squares_;
  std::atomic_uint_fast64_t buckets_[109]; // 109==BucketMapper::BucketCount()
  const uint64_t num_buckets_;
};

enum Operation {
  kRead,
  kWrite,
};

#endif  // ROCKSDB_UTILS_H
