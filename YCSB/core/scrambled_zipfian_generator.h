//
//  scrambled_zipfian_generator.h
//  YCSB-cpp
//
//  Copyright (c) 2020 Youngjae Lee <ls4154.lee@gmail.com>.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#ifndef YCSB_C_SCRAMBLED_ZIPFIAN_GENERATOR_H_
#define YCSB_C_SCRAMBLED_ZIPFIAN_GENERATOR_H_

#include "generator.h"

#include <iostream>
#include <unordered_map>
#include <cstdint>
#include <atomic>
#include "utils.h"
#include "zipfian_generator.h"

namespace ycsbc {

class ScrambledZipfianGenerator : public Generator<uint64_t> {
 public:
  ScrambledZipfianGenerator(uint64_t min, uint64_t max, double zipfian_const) :
      base_(min), num_items_(max - min + 1), generator_(0, 1000000000LL, zipfian_const) { }

  ScrambledZipfianGenerator(uint64_t min, uint64_t max) :
      base_(min), num_items_(max - min + 1),
      generator_(0, 1000000000LL, ZipfianGenerator::kZipfianConst) { }

  ScrambledZipfianGenerator(uint64_t num_items) :
      ScrambledZipfianGenerator(0, num_items - 1) { }

  uint64_t Next();

  uint64_t Last();

  void SetOperationCount(uint64_t operation_count) {
    operation_count_ = operation_count;
    if (operation_count_ && !prepared_) {
      Prepare();
      prepared_ = true;
    }
  }

 private:
  static constexpr double kZetan = 26.46902820178302;
  const uint64_t base_;
  const uint64_t num_items_;
  ZipfianGenerator generator_;

  bool                  prepared_ = false;
  uint64_t              operation_count_ = 0;
  std::atomic<int>      index_{0};
  std::vector<uint64_t> samples_;

  void Prepare();

  void CheckFreq();

  uint64_t Scramble(uint64_t value) const;
};

inline uint64_t ScrambledZipfianGenerator::Scramble(uint64_t value) const {
  return base_ + utils::FNVHash64(value) % num_items_;
}

inline uint64_t ScrambledZipfianGenerator::Next() {
  if (prepared_) {
    return base_ + samples_[index_++];
  } else {
    return Scramble(generator_.Next());
  }
}

inline uint64_t ScrambledZipfianGenerator::Last() {
  if (prepared_) {
    return base_ + samples_[index_];
  } else {
    return Scramble(generator_.Last());
  }
}

void ScrambledZipfianGenerator::Prepare() {
  std::cout << "Prepare data" << std::endl;
  samples_.resize(operation_count_);
  std::unordered_map<uint64_t, int> frequency;
  for (uint64_t i = 0; i < operation_count_; ++i) {
    samples_[i] = utils::FNVHash64(generator_.Next());
    frequency[samples_[i]]++;
  }

  std::unordered_map<uint64_t, uint64_t> mapper;
  std::vector<std::pair<int, uint64_t>> frequency_sorted;
  frequency_sorted.reserve(frequency.size());
  for (auto& pair : frequency) {
    frequency_sorted.emplace_back(pair.second, pair.first);
  }
  std::sort(frequency_sorted.begin(), frequency_sorted.end(),
            [](const std::pair<int, uint64_t>& p1, const std::pair<int, uint64_t>& p2){
              return p1.first > p2.first;
            });

  size_t idx = 0;
  size_t runs = (frequency.size() + num_items_ - 1) / num_items_;
  for (size_t run = 0; run < runs; ++run) {
    std::mt19937_64 shuffle_rd;
    srand((unsigned)time(0));
    shuffle_rd.seed(rand());
    std::vector<int> shuffled(num_items_);
    std::iota(std::begin(shuffled), std::end(shuffled), 0);
    std::shuffle(shuffled.begin(), shuffled.end(), shuffle_rd);

    for (size_t i = 0; i < num_items_ && idx < frequency.size(); ++i) {
      mapper[frequency_sorted[idx++].second] = shuffled[i];
    }
  }

  for (auto& sample : samples_) {
    sample = mapper[sample];
  }

  CheckFreq();
}

void ScrambledZipfianGenerator::CheckFreq() {
  std::unordered_map<uint64_t, int> freqs;
  for (auto& value : samples_) {
    ++freqs[value];
  }

  int cold_count = 0;
  for (auto& pair : freqs) {
    cold_count += (pair.second == 1);
  }

  int hot_count = freqs.size() - cold_count;
  float hot_freq = (samples_.size() - cold_count) / (float)samples_.size() * 100.f;
  printf("Sample generated. hot count = %d(%.2f), cold count = %d\n",
         hot_count, hot_freq, cold_count);
}

}

#endif // YCSB_C_SCRAMBLED_ZIPFIAN_GENERATOR_H_
