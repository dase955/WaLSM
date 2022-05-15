//
// Created by Chen Lixiang on 2022/3/3.
//

#pragma once

#include <unordered_map>

namespace ROCKSDB_NAMESPACE {

const int READ_AMP_BITS = 16;
const int RUNS_BITS = 16;
const int DATA_SIZE_BITS = 32;
const int READ_AMP_MASK = 0xFF;
const int RUNS_MASK = 0xFF;
const int DATA_SIZE_MASK = 0xFFFF;

const uint64_t INVALID_STATUS = 0xFFFFFFFFL;

struct QValue {
  double q_k, q_m;
  QValue(double q_k, double q_m) : q_k(q_k), q_m(q_m) {}
};

struct QState {
  std::vector<QValue> values_seq;
  QValue cur;
  QState(QValue val) : cur(val) {}
}; // State for q value records

class MergeQTable {
 public:
  // QTable constructor
  MergeQTable();

  // change current q table status
  void Jump(int, int, int);

  // update last status q value
  void Reward(int);

  // ask if current status should merge
  bool NeedsMerge();

 private:
  std::unordered_map<uint64_t , QValue> m_QTable;
  uint64_t m_cur_status;
  uint64_t m_last_status;
  int m_last_read_amp;
  const double learning_rate = 0.5;
  const double discount_rate = 0.5;
};
} // namespace ROCKSDB_NAMESPACE
