#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <cassert>
#include "macros.h"

namespace ROCKSDB_NAMESPACE {

struct SegmentAlgoInfo;
struct SegmentAlgoHelper;
class GreedyAlgo;

inline double StandardBenefit(const uint32_t& visit_cnt, const uint16_t& units_num);
inline double StandardCost(const uint32_t& visit_cnt, const uint16_t& units_num);
inline bool CompareSegmentAlgoHelper(const SegmentAlgoHelper& helper_1, const SegmentAlgoHelper& helper_2);

// contain visit counter of every segment in last long period
// also contain size of every segment's filter unit
// size of units belonging to one segment should be the same
// size equals to bits that one unit occupies
struct SegmentAlgoInfo {
    uint32_t visit_cnt;
    uint32_t size_per_unit;
    SegmentAlgoInfo(const uint32_t& cnt, const uint32_t& size) {
        assert(size > 0);
        visit_cnt = cnt; size_per_unit = size;
    }
};

// helper structure when performing this algo
// exactly, this structure will be the item of algo heap
struct SegmentAlgoHelper {
    uint32_t visit_cnt;
    uint16_t units_num;
    uint32_t size_per_unit;
    uint32_t segment_id;
    double enable_benifit; 
    SegmentAlgoHelper(const uint32_t& id, const uint32_t& cnt, const uint32_t& size, const uint16_t& units) {
        segment_id = id; visit_cnt = cnt; size_per_unit = size; units_num = units;
        enable_benifit = StandardBenefit(visit_cnt, units_num);
        // assert(units_num <= MAX_UNITS_NUM);
    }
    SegmentAlgoHelper(const uint32_t& id, SegmentAlgoInfo& segment_algo_info) {
        segment_id = id; visit_cnt = segment_algo_info.visit_cnt; 
        size_per_unit = segment_algo_info.size_per_unit; units_num = 0;
        enable_benifit = StandardBenefit(visit_cnt, units_num);
    }
};

inline double StandardBenefit(const uint32_t& visit_cnt, const uint16_t& units_num) {
    int bits_per_key = BITS_PER_KEY_PER_UNIT;
    // We intentionally round down to reduce probing cost a little bit
    int num_probes = static_cast<int>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
    if (num_probes < 1) num_probes = 1;
    if (num_probes > 30) num_probes = 30;
        
    // compute false positive rate of one filter unit
    double rate_per_unit = std::pow(1.0 - std::exp(-double(num_probes) / double(bits_per_key)), num_probes);

    if (units_num >= MAX_UNITS_NUM) {
        return 0.0;
    }

    uint16_t next_units_num = units_num + 1;
    double rate = std::pow(rate_per_unit, units_num);
    double next_rate = std::pow(rate_per_unit, next_units_num);

    double benefit = double(visit_cnt) * (rate - next_rate);
    /*
    std::cout << "visit_cnt : " << visit_cnt
                << " , rate : " << rate
                << " , next_rate : " << next_rate
                << " . rate_per_unit : " << rate_per_unit 
                << std::endl;
    */
    assert(benefit >= 0);
    return benefit;
}

inline double StandardCost(const uint32_t& visit_cnt, const uint16_t& units_num) {
    int bits_per_key = BITS_PER_KEY_PER_UNIT;
    // We intentionally round down to reduce probing cost a little bit
    int num_probes = static_cast<int>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
    if (num_probes < 1) num_probes = 1;
    if (num_probes > 30) num_probes = 30;
        
    // compute false positive rate of one filter unit
    double rate_per_unit = std::pow(1.0 - std::exp(-double(num_probes) / double(bits_per_key)), num_probes);

    if (units_num <= MIN_UNITS_NUM) {
        return __DBL_MAX__;
    }

    uint16_t next_units_num = units_num - 1;
    double rate = std::pow(rate_per_unit, units_num);
    double next_rate = std::pow(rate_per_unit, next_units_num);

    double cost = double(visit_cnt) * (next_rate - rate);
    /*
    std::cout << "visit_cnt : " << visit_cnt
                << " , rate : " << rate
                << " , next_rate : " << next_rate
                << " . rate_per_unit : " << rate_per_unit 
                << std::endl;
    */
    assert(cost >= 0);
    return cost;
}

inline bool CompareSegmentAlgoHelper(const SegmentAlgoHelper& helper_1, const SegmentAlgoHelper& helper_2) {
    return helper_1.enable_benifit < helper_2.enable_benifit;
}

class GreedyAlgo {
public: 
    // segment_algo_infos: map<segment id, SegmentAlgoInfo>
    // algo_solution: map<segment id, units num>
    // cache_size: total size of filter cache, we can left some space for emergency needs
    // we can input 95% of real total size as arg cache_size, then left 5% of space for further needs
    // full debug process of GreedyAlgo, not thread-secured
    // so make sure that only called by one thread
    void solve(std::map<uint32_t, SegmentAlgoInfo>& segment_algo_infos,
                std::map<uint32_t, uint16_t>& algo_solution, const uint32_t& cache_size);
    // full debug process of GreedyAlgo, not thread-secured
    // so make sure that only called by one thread
    void debug(std::map<uint32_t, uint16_t>& algo_solution, const uint32_t& cache_size) {
        // generate debug data
        std::map<uint32_t, SegmentAlgoInfo> segment_algo_infos;
        segment_algo_infos.clear();
        uint32_t min_segment_id = 0, max_segment_id = 9999;
        for (uint32_t segment_id = min_segment_id; segment_id <= max_segment_id; segment_id++) {
            // SegmentAlgoInfo segment_algo_info(segment_id * 1000, 8 * 1024 * 8); // one unit is 8kb
            // segment_algo_infos[segment_id] = SegmentAlgoInfo(segment_id * 1000, 8 * 1024 * 8);
            // directly use '=' will cause bug, try use std::map.insert
            segment_algo_infos.insert(std::make_pair(segment_id, 
                                        SegmentAlgoInfo(segment_id * std::pow(10, (segment_id / 3000) + 1), 8 * 1024 * 4)));  // one unit 2 kb
        }
        assert(segment_algo_infos.size() == max_segment_id + 1);

        // already generate debug data, try perform algo
        solve(segment_algo_infos, algo_solution, cache_size);

        // simple check results
        // noticed that if segment a visit_cnt >= segment b visit_cnt
        // then segment a units_num >= segment b units_num
        for (uint32_t segment_id = min_segment_id; segment_id < max_segment_id; segment_id++) {
            assert(algo_solution[segment_id] <= algo_solution[segment_id + 1]);
        }
    }
};


}