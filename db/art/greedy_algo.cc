#include "greedy_algo.h"
#include <cassert>
#include <set>
#include <fstream>
#include <iostream>

namespace ROCKSDB_NAMESPACE {

// this func is not thread-secured, so make only one thread perform this algo!!!
void GreedyAlgo::solve(std::map<uint32_t, SegmentAlgoInfo>& segment_algo_infos,
                        std::map<uint32_t, uint16_t>& algo_solution, const uint32_t& cache_size) {
    assert(!segment_algo_infos.empty());
    // ready to perform algo
    algo_solution.clear();
    std::vector<SegmentAlgoHelper> segment_algo_helper_heap;
    for (auto it = segment_algo_infos.begin(); it != segment_algo_infos.end(); it++) {
        uint32_t segment_id = it->first;
        SegmentAlgoInfo segment_algo_info = it->second;
        algo_solution[segment_id] = 0; // init algo_solution

        SegmentAlgoHelper segment_algo_helper(segment_id, segment_algo_info);
        segment_algo_helper_heap.emplace_back(segment_algo_helper); // init algo heap
    }
    assert(segment_algo_infos.size() == algo_solution.size());
    assert(segment_algo_infos.size() == segment_algo_helper_heap.size());
    std::make_heap(segment_algo_helper_heap.begin(),
                    segment_algo_helper_heap.end(), 
                    CompareSegmentAlgoHelper);  

    std::fstream f_algo;
    f_algo.open("/pg_wal/ycc/algo.log", std::ios::out | std::ios::app);
    f_algo << "[DEBUG] start to record algo : " << std::endl;

    // current used space size (bits) of filter cache
    uint32_t current_cache_size = 0; 
    while (!segment_algo_helper_heap.empty()) {
        // std::cout << "segment id : " << segment_algo_helper_heap[0].segment_id << std::endl;

        const size_t size = segment_algo_helper_heap.size();
        // heap top item moved to segment_algo_helper_heap[segment_algo_helper_heap.size()-1];
        std::pop_heap(segment_algo_helper_heap.begin(),
                        segment_algo_helper_heap.end(),
                        CompareSegmentAlgoHelper);
        SegmentAlgoHelper segment_algo_helper_top = segment_algo_helper_heap[size-1];
        // check whether free space (in filter cache) is enough
        uint32_t size_needed = segment_algo_helper_top.size_per_unit;
        // if not enough, remove this segment helper from heap
        // that means we will not consider this segment any longer
        if (current_cache_size + size_needed > cache_size) {
            segment_algo_helper_heap.pop_back();
            continue;
        }
        // SegmentAlgoHelper(const uint32_t& id, const uint32_t& cnt, const uint32_t& size, const uint16_t& units)
        SegmentAlgoHelper segment_algo_helper_needed(segment_algo_helper_top.segment_id,
                                                        segment_algo_helper_top.visit_cnt,
                                                        segment_algo_helper_top.size_per_unit,
                                                        segment_algo_helper_top.units_num + 1);
        // update enabled units
        // noticed that if one segment visit cnt == 0, it still enable one unit
        // so check visit num before update algo_solution
        if (segment_algo_helper_needed.visit_cnt > 0) {
            algo_solution[segment_algo_helper_needed.segment_id] = segment_algo_helper_needed.units_num;
            current_cache_size += size_needed;
            f_algo << "[DEBUG] segment " << segment_algo_helper_needed.segment_id
                    << " : " << segment_algo_helper_needed.units_num - 1 << " -> "
                    << segment_algo_helper_needed.units_num << " , cache space left : " 
                    << cache_size - current_cache_size << " , recv benefit : " 
                    << segment_algo_helper_top.enable_benifit << " , next benefit : " 
                    << segment_algo_helper_needed.enable_benifit << std::endl;
        }
        assert(algo_solution[segment_algo_helper_needed.segment_id] <= MAX_UNITS_NUM);
        // enable benefit == 0 means units_num == MAX_UNITS_NUM
        // that means we cannot enable one unit for this segment, already enable all units
        if (segment_algo_helper_needed.enable_benifit == 0) {
            // assert(segment_algo_helper_needed.units_num >= MAX_UNITS_NUM);
            segment_algo_helper_heap.pop_back();
            continue;
        }
        // we can push this new segment helper into heap
        segment_algo_helper_heap[size-1] = segment_algo_helper_needed;
        std::push_heap(segment_algo_helper_heap.begin(),
                        segment_algo_helper_heap.end(),
                        CompareSegmentAlgoHelper);
    }

    f_algo << std::endl;
    f_algo.close();
    // return nothing, all results should be written into algo_solution
}

}