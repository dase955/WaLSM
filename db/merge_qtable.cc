//
// Created by Chen Lixiang on 2022/3/3.
//

#include "db/merge_qtable.h"
#include <cmath>
#include <random>

namespace ROCKSDB_NAMESPACE {

  MergeQTable::MergeQTable() : m_cur_status(INVALID_STATUS),
                             m_last_status(INVALID_STATUS),
                             m_last_read_amp(0){ }

  void MergeQTable::Jump(int read_amp, int runs, int data_size) {
    // construct q value
    uint64_t new_status =
        ((uint64_t) (read_amp & READ_AMP_MASK) << READ_AMP_BITS) +
        ((uint64_t) (runs & RUNS_MASK) << RUNS_BITS) +
        ((uint64_t) (data_size & DATA_SIZE_MASK) << DATA_SIZE_BITS);

    m_last_status = m_cur_status;

    if (m_QTable.find(new_status) == m_QTable.end()) {
      // initialize current status q values
      m_QTable[new_status] = QValue(0.9, 0.1);
    }
    m_cur_status = new_status;
    m_last_read_amp = read_amp;
  }

  void MergeQTable::Reward(int read_amp) {
    if (m_cur_status == INVALID_STATUS || m_last_status == INVALID_STATUS) {
      return;
    }

    // double r = exp((double) read_amp);
    if (m_cur_status == m_last_status) {
      // last action is keep
      m_QTable[m_last_status].q_k += discount_rate * read_amp;
    } else {
      // last action is merge
      m_QTable[m_last_status].q_m += discount_rate * read_amp;
    }
  }

  bool MergeQTable::NeedsMerge() {
    if (m_cur_status == INVALID_STATUS) {
      return false;
    }
    QValue &qv = m_QTable.at(m_cur_status);
    double score_k = exp(qv.q_k), score_m = exp(qv.q_m);
    // formalize
    double prob_k = score_k / (score_k + score_m);
    // generate random number in (0, 1]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double rn = dis(gen);

    if (rn < prob_k) {
      // do not merge
      return false;
    }
    return true;
  }

} // namespace ROCKSDB_NAMESPACE
