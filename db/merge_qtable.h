//
// Created by Chen Lixiang on 2022/3/3.
//

#pragma once

#include <unordered_map>
#include <vector>
#include <random>

namespace ROCKSDB_NAMESPACE {

const int READ_AMP_BITS = 16;
const int RUNS_BITS = 16;
const int DATA_SIZE_BITS = 32;
const int READ_AMP_MASK = 0xFF;
const int RUNS_MASK = 0xFF;
const int DATA_SIZE_MASK = 0xFFFF;
const uint64_t INVALID_STATUS = 0xFFFFFFFFL;

const uint64_t DATA_SIZE_ROUND = 64 * 1024 * 1024L;
const uint64_t READ_PENALTY_ROUND = 1e9L;

/**
 *
 * +-----------+--------------+-------------+
 * | DATA_SIZE | READ_PENALTY | SORTED_RUNS |
 * +-----------+--------------+-------------+
 * |   32bit   |     16bit    |     16bit   |
 * +-----------+--------------+-------------+
 *
 */

struct QValue {
  double q_k, q_m;
  QValue(double q_keep_, double q_merge_) : q_k(q_keep_), q_m(q_merge_) {}
};

struct QKey {
  uint64_t q_state;
  uint64_t penalty;
  double reward;
  bool keep;
  QKey(uint64_t state_, uint64_t penalty_, double reward_, bool keep_)
      : q_state(state_), penalty(penalty_), reward(reward_), keep(keep_) {}
};

class MergeQTable {
 public:
  // QTable constructor
  MergeQTable(double alpha, double gamma)
      : learning_rate(alpha), discount_rate(gamma) {}

  static uint64_t genQState(uint64_t data_size, uint64_t penalty, size_t runs) {
    data_size /= DATA_SIZE_ROUND;
    penalty /= READ_PENALTY_ROUND;
    return (data_size << DATA_SIZE_BITS) + (penalty << READ_AMP_BITS) +
           static_cast<uint64_t>(runs);
  }

  // update last status q value
  void Reward(std::vector<QKey>& keys, bool is_tier) {
    auto& table = is_tier ? tier_table : level_table;
    double gt = 0;
    for (int i = static_cast<int>(keys.size()) - 1; i >= 0; i--) {
      gt *= gt;
      gt += keys[i].reward;
      if (table.find(keys[i].q_state) == table.end()) {
        table.insert({keys[i].q_state, QValue(50, 50)});
      }

      auto kv = table.find(keys[i].q_state);
      if (keys[i].keep) {
        if (gt > 0) kv->second.q_k += gt;
        else kv->second.q_m += abs(gt);
      } else {
        if (gt > 0) kv->second.q_m += gt;
        else kv->second.q_m += abs(gt);
      }
    }
  }

  // ask if current status should merge
  bool ShouldMerge(uint64_t state, bool is_tier) {
    auto& table = is_tier ? tier_table : level_table;
    auto it = table.find(state);
    if (it == table.end()) {
      table.insert({state, QValue{50, 50}});
      it = table.find(state);
    }
    auto& value = it->second;
    double prob = value.q_m / (value.q_m + value.q_k);
    return random_bool(prob);
  }

  bool random_bool(double prob) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double rn = dis(gen);
    return rn < prob;
  }

  std::unordered_map<uint64_t, QValue> level_table;
  std::unordered_map<uint64_t, QValue> tier_table;
  int m_last_read_amp;
  const double learning_rate = 0.5;
  const double discount_rate = 0.5;
};
}  // namespace ROCKSDB_NAMESPACE
