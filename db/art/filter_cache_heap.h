#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <cassert>
#include <mutex>
#include "macros.h"

namespace ROCKSDB_NAMESPACE {
// when filter cache is full , 
// we need to use heap manager to 
// clear some space and insert new filter units 
// for these coming new segments
// or we may need to use heap manager to adjust filter cache
// to reduce extra I/O caused by false positive 

struct FilterCacheHeapItem;
typedef FilterCacheHeapItem* FilterCacheHeapNode;
struct FilterCacheModifyResult;
class FilterCacheHeap;
class FilterCacheHeapManager;
inline bool FilterCacheHeapNodeLessComparor(const FilterCacheHeapNode& node_1, const FilterCacheHeapNode& node_2);
inline bool FilterCacheHeapNodeGreaterComparor(const FilterCacheHeapNode& node_1, const FilterCacheHeapNode& node_2);
inline double StandardBenefitWithMaxBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& max_bound);
inline double StandardCostWithMinBound(const uint32_t& visit_cnt, const uint16_t& units_num, const uint16_t& min_bound);

struct FilterCacheHeapItem {
    uint32_t segment_id;
    uint32_t approx_visit_cnt; // estimated visit cnt
    uint16_t current_units_num; // enabled units num for this segment
    double benefit_or_cost; // can represent enable benefit or disable cost
    uint16_t units_num_limit; // units num prediction model predict maximum units num for every segment
    bool is_alive; // sign whether this item still used, if false, that means this segment already merged and freed
    // default set heap_value = 0, we will compuate benefit or cost in batch upsert func
    FilterCacheHeapItem(const uint32_t& id, const uint32_t& cnt, const uint16_t& units, const double& heap_value, const uint16_t& limit) {
        segment_id = id; 
        approx_visit_cnt = cnt; 
        current_units_num = units; 
        benefit_or_cost = heap_value; 
        units_num_limit = limit; 
        is_alive = true;
        assert(current_units_num >= MIN_UNITS_NUM);
        assert(current_units_num <= units_num_limit);
    }
    /*
    FilterCacheHeapItem(const FilterCacheHeapItem& item) {
        segment_id = item.segment_id; 
        approx_visit_cnt = item.approx_visit_cnt; 
        current_units_num = item.current_units_num; 
        benefit_or_cost = item.benefit_or_cost; 
        units_num_limit = item.units_num_limit;
        is_alive = item.is_alive;
    }
    */
};

struct FilterCacheModifyResult {
    uint32_t enable_segment_id;
    uint32_t disable_segment_id;
    uint16_t enable_segment_units_num;
    uint16_t disable_segment_units_num;
    uint16_t enable_segment_next_units_num;
    uint16_t disable_segment_next_units_num;
    double enable_benefit;
    double disable_cost;
};

inline bool FilterCacheHeapNodeLessComparor(const FilterCacheHeapNode& node_1, const FilterCacheHeapNode& node_2) {
    return node_1->benefit_or_cost < node_2->benefit_or_cost;
}

inline bool FilterCacheHeapNodeGreaterComparor(const FilterCacheHeapNode& node_1, const FilterCacheHeapNode& node_2) {
    return node_1->benefit_or_cost > node_2->benefit_or_cost;
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

class FilterCacheHeap {
private:
    int heap_type_;
    // map<segment id, node>, use this map to fastly locate node in heap
    std::map<uint32_t, FilterCacheHeapNode> heap_index_; 
    // use make_heap, push_heap, pop_heap to manage heap
    std::vector<FilterCacheHeapNode> heap_; 
    // std::mutex heap_mutex_;

public:
    FilterCacheHeap() {
        heap_type_ = UNKNOWN_HEAP;
        heap_index_.clear();
        heap_.clear();
    }

    ~FilterCacheHeap() {
        // do nothing
    }

    void set_type(const int type) {
        heap_type_ = type;
    }

    // only rebuild heap_, do nothing to heap_index_
    void rebuild_heap() {
        // heap_mutex_.lock();

        assert(heap_type_ != UNKNOWN_HEAP);
        assert(heap_.size() == heap_index_.size());
        if (heap_type_ == BENEFIT_HEAP) {
            std::make_heap(heap_.begin(), heap_.end(), FilterCacheHeapNodeLessComparor);
        } else if (heap_type_ == COST_HEAP) {
            std::make_heap(heap_.begin(), heap_.end(), FilterCacheHeapNodeGreaterComparor);
        }

        // heap_mutex_.unlock();
    }

    // return heap top
    FilterCacheHeapNode heap_top();

    // pop one node with deleting node from heap_index_
    // void pop();

    // push one node with upsert node into heap_index_
    // void push(FilterCacheHeapNode& node);

    // given a batch of segment id, return needed nodes. 
    // only support batch query and reminded that one return node may be null 
    // if segment not available or segment not exists in heap_index_
    // result will write into return_nodes
    void batch_query(std::vector<uint32_t>& segment_ids, std::vector<FilterCacheHeapNode>& return_nodes);

    // upsert batch nodes into heap_index_ and heap_
    // only support batch upsert, if one node already exists in heap_index_, it must in heap
    // so we only need to update the content of that existing node
    void batch_upsert(std::vector<FilterCacheHeapNode>& nodes);

    // delete batch nodes from heap_index_ and heap_
    // only support batch delete, if one node not exist in heap_index_, it must not exist in heap
    // so we only need to delete these existing nodes
    void batch_delete(std::vector<uint32_t>& segment_ids);

    // only used in debug !!!
    void heap_index(std::map<uint32_t, FilterCacheHeapNode>& heap_index) {
        heap_index.clear();
        heap_index.insert(heap_index_.begin(), heap_index_.end());
    }

    // only used in debug !!!
    void heap(std::vector<FilterCacheHeapNode>& heap) {
        heap.clear();
        heap.assign(heap_.begin(), heap_.end());
    }
};

class FilterCacheHeapManager {
private: 
    static FilterCacheHeap benefit_heap_;
    static FilterCacheHeap cost_heap_;
    // set heap node visit cnt = c_1, real estimated visit cnt = c_2
    // we only update c_1 when | c_1 - c_2 | >= VISIT_CNT_UPDATE_BOUND
    // update c_1 means we need to update this recorder and heap
    // heap_visit_cnt_recorder: map<segment id : visit cnt in heap>
    // when filter cache call delete, this recorder will automately delete these merged segment ids
    // when filter cache call upsert, this recorder will automately upsert these segment ids
    static std::map<uint32_t, uint32_t> heap_visit_cnt_recorder_;
    static std::map<uint32_t, uint16_t> units_num_limit_recorder_;
    // TODO: mutex can be optimized
    static std::mutex manager_mutex_; 

public:
    FilterCacheHeapManager() {
        benefit_heap_.set_type(BENEFIT_HEAP);
        cost_heap_.set_type(COST_HEAP);
        heap_visit_cnt_recorder_.clear();
        units_num_limit_recorder_.clear();
    }

    ~FilterCacheHeapManager() {
        // do nothing
    }

    // sync units_num_limit in heap and recorder
    // reminded that we will not insert or delete nodes in this method
    // we only update these nodes that already exist in two heaps
    void sync_units_num_limit(std::map<uint32_t, uint16_t>& current_units_num_limit_recorder);

    // sync visit cnt in heap and real estimated visit cnt
    // reminded that we will not insert or delete nodes in this method
    // we only update these nodes that already exist in two heaps
    void sync_visit_cnt(std::map<uint32_t, uint32_t>& current_visit_cnt_recorder);

    // try to read benefit_heap top and cost_heap top, then judge whether we need to modify units num in filter cache
    // return true when we can modify units num of several segments, return false when we cannot
    // reminded that this func only modify heap, we still need to update filter units in filter cache
    bool try_modify(FilterCacheModifyResult& result);

    // delete batch segment nodes in benefit_heap and cost_heap, also need to update heap_visit_cnt_recorder_
    void batch_delete(std::vector<uint32_t>& segment_ids);

    // upsert batch segment nodes in benefit_heap and cost_heap, also need to update heap_visit_cnt_recorder_
    // only input items, we will allocate space for nodes later
    // reminded that we will also update heap_visit_cnt_recorder_ if we update a existed node
    // because we need to keep heap visit cnt and recorder visit cnt the same
    void batch_upsert(std::vector<FilterCacheHeapItem>& items);

    // 1. try debug batch insert
    // 2. try debug batch update(use batch_upsert)
    void debug();
};

}
