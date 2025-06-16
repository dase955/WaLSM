#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>
#include <cassert>
#include <mutex>
#include "macros.h"

namespace ROCKSDB_NAMESPACE {

class FilterCacheIntervals;
class FilterCacheIntervalsQueue;
class FilterCacheIntervalsQueuesManager;
struct FilterCacheModifyResult;

inline bool FilterCacheBenefitComparor(const FilterCacheIntervals& node_1, const FilterCacheIntervals& node_2);
inline bool FilterCacheCostComparor(const FilterCacheIntervals& node_1, const FilterCacheIntervals& node_2);
inline double StandardBenefitWithMaxBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& max_bound);
inline double StandardCostWithMinBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& min_bound);
inline bool CheckGreaterThanAlpha(const double& left_benefit, const double& right_benefit, 
                           const double& left_cost, const double& right_cost);

struct FilterCacheModifyResult {
    uint32_t enable_segment_id;
    uint32_t disable_segment_id;
    uint16_t enable_units_num;
    uint16_t disable_units_num;
    uint16_t enable_units_limit;
    uint16_t disable_units_limit;
};

inline bool CheckGreaterThanAlpha(const double& left_benefit, const double& right_benefit, 
                           const double& left_cost, const double& right_cost) {
    if (left_benefit >= right_cost) { // status 1
        return true;
    } else if (left_cost >= right_benefit) { // status 3
        return false;
    } else if (left_cost >= left_benefit && right_cost >= right_benefit) { // status 2.4
        return false;
    } else if (left_cost < left_benefit && right_cost < right_benefit) { // status 2.1
        return (0.5 * (right_cost - left_benefit) * (right_cost - left_benefit) < 
                (1 - ALPHA) * (right_benefit - left_benefit) * (right_cost - left_cost));
    } else if (left_cost < left_benefit && right_cost >= right_benefit) { // status 2.2
        return ((1 - ALPHA) * left_cost + ALPHA * right_cost <
                0.5 * (left_benefit + right_benefit));
    } else if (left_cost >= left_benefit && right_cost < right_benefit) { // status 2.3
        return (ALPHA * left_benefit + (1 - ALPHA) * right_benefit >
                0.5 * (left_cost + right_cost));
    }
} 

inline double StandardBenefitWithMaxBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& max_bound) {
    int bits_per_key = BITS_PER_KEY_PER_UNIT;
    // We intentionally round down to reduce probing cost a little bit
    int num_probes = static_cast<int>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
    if (num_probes < 1) num_probes = 1;
    if (num_probes > 30) num_probes = 30;
        
    // compute false positive rate of one filter unit
    double rate_per_unit = std::pow(1.0 - std::exp(-double(num_probes) / double(bits_per_key)), num_probes);

    assert(max_bound >= MIN_UNITS_NUM);
    assert(max_bound <= MAX_UNITS_NUM);
    if (units_num >= max_bound) { 
        return 0.0; // 0.0 is the lowest value of benefit (benefit >= 0.0)
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

inline double StandardCostWithMinBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& min_bound) {
    int bits_per_key = BITS_PER_KEY_PER_UNIT;
    // We intentionally round down to reduce probing cost a little bit
    int num_probes = static_cast<int>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
    if (num_probes < 1) num_probes = 1;
    if (num_probes > 30) num_probes = 30;
        
    // compute false positive rate of one filter unit
    double rate_per_unit = std::pow(1.0 - std::exp(-double(num_probes) / double(bits_per_key)), num_probes);

    assert(min_bound >= MIN_UNITS_NUM);
    assert(min_bound <= MAX_UNITS_NUM);
    if (units_num <= min_bound) {
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

class FilterCacheIntervals {
    public:
        std::vector<uint32_t> segment_ids;

        double left_val;

        double right_val;

        uint16_t units_limit; // units num prediction model predict maximum units num for every segment

        uint16_t units_curr;

        bool is_benefit;

        FilterCacheIntervals(const uint16_t& units, const uint16_t& limit, const bool& benefit) {
            units_curr = units;
            units_limit = limit; 
            is_benefit = benefit;
            left_val = 0;
            right_val = 0;
            segment_ids.clear();
        }

        void Insert(const uint32_t& segment_id) { segment_ids.push_back(segment_id); };

        uint32_t Pop() { 
            uint32_t res = segment_ids[0];
            segment_ids.erase(segment_ids.begin()); 
            return res;
        };

        void clear() { segment_ids.clear(); }

        void Remove(const uint32_t& segment_id) { std::remove(segment_ids.begin(), segment_ids.end(), segment_id); }

        void Update(const uint32_t& left_cnt, const uint32_t& right_cnt) {
            if (is_benefit) {
                left_val = StandardBenefitWithMaxBound(left_cnt, units_curr, units_limit);
                right_val = StandardBenefitWithMaxBound(right_cnt, units_curr, units_limit);
            } else {
                left_val = StandardCostWithMinBound(left_cnt, units_curr, 0);
                right_val = StandardCostWithMinBound(right_cnt, units_curr, 0);
            }
        }

        bool empty() { return segment_ids.empty(); }

        double left() { return left_val; }

        double right() { return right_val; }
};

class FilterCacheIntervalsQueue {
    public: 
        std::vector<FilterCacheIntervals> queue;
        std::map<uint32_t, uint16_t> index;
        bool is_benefit;

        FilterCacheIntervalsQueue(const bool& benefit) {
            queue.clear();
            index.clear();
            is_benefit = benefit;
            for (uint16_t limit = MIN_UNITS_NUM; limit <= MAX_UNITS_NUM; limit++) {
                for (uint16_t units = MIN_UNITS_NUM; units <= limit; units++) {
                    queue.push_back(FilterCacheIntervals(units, limit, benefit));
                }
            }
        } 

        void Update(std::map<uint16_t, uint32_t>& min_cnts, std::map<uint16_t, uint32_t>& max_cnts) {
            for (int i = 0; i < queue.size(); i++) {
                const uint32_t left_cnt = min_cnts[queue[i].units_limit];
                const uint32_t right_cnt = max_cnts[queue[i].units_limit];
                queue[i].Update(left_cnt, right_cnt);
            }
            if (is_benefit)
                std::sort(queue.begin(), queue.end(), FilterCacheBenefitComparor);
            else 
                std::sort(queue.begin(), queue.end(), FilterCacheCostComparor);
        }

        void AddSegment(const uint32_t& segment_id, const uint16_t& segment_units, const uint16_t& segment_limit) {
            int i = 0;
            for (uint16_t limit = MIN_UNITS_NUM; limit <= MAX_UNITS_NUM; limit++) {
                for (uint16_t units = MIN_UNITS_NUM; units <= limit; units++) {
                    if (segment_units == units && segment_limit == limit) {
                        break;
                    } else {
                        i++;
                    }
                }
            }
            if (i < queue.size()) {
                queue[i].Insert(segment_id);
                index[segment_id] = i;
            }
        }

        void clear() {
            for (int i = 0; i < queue.size(); i++) {
                queue[i].clear();
            }
        }

        void DelSegment(const uint32_t& segment_id) {
            queue[index[segment_id]].Remove(segment_id);
            index.erase(segment_id);
        }

        uint32_t PopSegment() {
            uint32_t segment_id = queue[index[segment_id]].Pop();
            return segment_id;
        }
};
    
class FilterCacheIntervalsQueuesManager {
    public:
        FilterCacheIntervalsQueue *benefit_queue;

        FilterCacheIntervalsQueue *cost_queue;

        std::set<uint32_t> valid_segment_ids;

        std::map<uint32_t, uint16_t> units_saver, limit_saver;

        std::mutex lock;
    
        FilterCacheIntervalsQueuesManager() {
            benefit_queue = new FilterCacheIntervalsQueue(true);
            cost_queue = new FilterCacheIntervalsQueue(false);
        }

        void UpdateCnt(std::map<uint16_t, uint32_t>& min_cnts, std::map<uint16_t, uint32_t>& max_cnts) {
            lock.lock();
            units_saver.clear(); limit_saver.clear();
            benefit_queue->Update(min_cnts, max_cnts);
            cost_queue->Update(min_cnts, max_cnts);
            lock.unlock();
        }

        void UpdateLimit(std::map<uint32_t, uint16_t>& unit_limits) {
            lock.lock();
            units_saver.clear(); limit_saver.clear();
            benefit_queue->clear();
            cost_queue->clear();
            auto it = unit_limits.begin();
            for (uint16_t segment_id : valid_segment_ids) {
                it = unit_limits.find(segment_id);
                if (it != unit_limits.end()) {
                    UpdSegment(segment_id, it->second, it->second);
                }
            }
            lock.unlock();
        }

        void AddSegment(const uint32_t& segment_id, const uint16_t& segment_units, const uint16_t& segment_limit) {
            lock.lock();
            benefit_queue->AddSegment(segment_id, segment_units, segment_limit);
            cost_queue->AddSegment(segment_id, segment_units, segment_limit);
            valid_segment_ids.insert(segment_id);
            lock.unlock();
        }

        void UpdSegment(const uint32_t& segment_id, const uint16_t& segment_units, const uint16_t& segment_limit) {
            //lock.lock();
            if (valid_segment_ids.count(segment_id)) {
                benefit_queue->AddSegment(segment_id, segment_units, segment_limit);
                cost_queue->AddSegment(segment_id, segment_units, segment_limit);
            }
            //lock.unlock();
        }

        void DelSegment(const uint32_t& segment_id) {
            valid_segment_ids.erase(segment_id);
            lock.lock();
            benefit_queue->DelSegment(segment_id);
            cost_queue->DelSegment(segment_id);
            units_saver.erase(segment_id);
            limit_saver.erase(segment_id);
            lock.unlock();
        }

        int SearchCostQueue(int benefit_i) {
            double left_benefit = benefit_queue->queue[benefit_i].left_val;
            double right_benefit = benefit_queue->queue[benefit_i].right_val;
            int i = 0;
            while (i < cost_queue->queue.size()) {
                double left_cost = cost_queue->queue[i].left_val;
                double right_cost = cost_queue->queue[i].right_val;
                if (CheckGreaterThanAlpha(left_benefit, right_benefit, left_cost, right_cost)
                    && !cost_queue->queue[i].empty()) {
                    return i;
                } else {
                    i++;
                }
            }
            return cost_queue->queue.size();
        }

        FilterCacheModifyResult replace(int benefit_i, int cost_i) {
            FilterCacheModifyResult result;

            result.disable_segment_id = cost_queue->queue[cost_i].Pop();
            result.disable_units_limit = cost_queue->queue[cost_i].units_limit;
            result.disable_units_num = cost_queue->queue[cost_i].units_curr - 1;
            result.enable_segment_id = benefit_queue->queue[benefit_i].Pop();
            result.enable_units_limit = benefit_queue->queue[benefit_i].units_limit;
            result.enable_units_num = benefit_queue->queue[benefit_i].units_curr + 1;

            return result;
        }

        bool adjust(FilterCacheModifyResult& result) {
            int benefit_i = 0;
            int cost_i = 0;
            bool found = false;
            for (benefit_i = 0; benefit_i < benefit_queue->queue.size(); benefit_i++) {
                while (!benefit_queue->queue[benefit_i].empty()) {
                    lock.lock();
                    cost_i = SearchCostQueue(benefit_i);
                    if (cost_i == cost_queue->queue.size()) {
                        uint32_t segment_id = benefit_queue->PopSegment();
                        units_saver[segment_id] = benefit_queue->queue[benefit_i].units_curr;
                        limit_saver[segment_id] = benefit_queue->queue[benefit_i].units_limit;
                    } else {
                        result = replace(benefit_i, cost_i);
                        found = true;
                    }
                    lock.unlock();

                    if (found) {
                        lock.lock();
                        if (valid_segment_ids.count(result.enable_segment_id)) {
                            UpdSegment(result.enable_segment_id, result.enable_units_num, result.enable_units_limit);
                        }
                        if (valid_segment_ids.count(result.disable_segment_id)) {
                            UpdSegment(result.disable_segment_id, result.disable_units_num, result.disable_units_limit);
                        }
                        lock.unlock();
                        return true;
                    }
                }
            }

            if (!found) {
                lock.lock();
                auto units_it = units_saver.begin();
                while (units_it != units_saver.end()) {
                    if (valid_segment_ids.count(units_it->first)) {
                        UpdSegment(units_it->first, units_it->second, limit_saver[units_it->first]);
                    }
                    units_it++;
                }
                lock.unlock();
                units_saver.clear();
                limit_saver.clear();
            }

            return false;
        }
};

inline bool FilterCacheBenefitComparor(const FilterCacheIntervals& node_1, const FilterCacheIntervals& node_2) {
    return node_1.right_val > node_2.right_val;
}

inline bool FilterCacheCostComparor(const FilterCacheIntervals& node_1, const FilterCacheIntervals& node_2) {
    return node_1.left_val < node_2.left_val;
}

}

