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

    // std::fstream f_algo;
    // f_algo.open("/home/guoteng_20241228_135/WaLSM+/log/algo.log", std::ios::out | std::ios::app);
    // f_algo << "[DEBUG] start to record algo : " << std::endl;

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
        // if not enough, exit. we allocate the same size to all units.
        if (current_cache_size + size_needed > cache_size) {
            break;
        }
        // SegmentAlgoHelper(const uint32_t& id, const uint32_t& cnt, const uint32_t& size, const uint16_t& units)
        SegmentAlgoHelper segment_algo_helper_needed(segment_algo_helper_top.segment_id,
                                                        segment_algo_helper_top.visit_cnt,
                                                        segment_algo_helper_top.size_per_unit,
                                                        segment_algo_helper_top.units_num + 1);
        // update enabled units
        // noticed that if one segment visit cnt == 0, it enable zero units
        // so check visit num before update algo_solution
        if (segment_algo_helper_needed.visit_cnt > 0) {
            algo_solution[segment_algo_helper_needed.segment_id] = segment_algo_helper_needed.units_num;
            current_cache_size += size_needed;
            // f_algo << "[DEBUG] segment " << segment_algo_helper_needed.segment_id
            //         << " : " << segment_algo_helper_needed.units_num - 1 << " -> "
            //         << segment_algo_helper_needed.units_num << " , cache space left : " 
            //         << cache_size - current_cache_size << " , recv benefit : " 
            //         << segment_algo_helper_top.enable_benifit << " , next benefit : " 
            //         << segment_algo_helper_needed.enable_benifit << " , visit count: "
            //         << segment_algo_helper_needed.visit_cnt << std::endl;
        }
        assert(algo_solution[segment_algo_helper_needed.segment_id] <= MAX_UNITS_NUM);
        // enable benefit == 0 means units_num == MAX_UNITS_NUM or its visit cnt == 0
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

    // f_algo << std::endl;
    // f_algo.close();
    // return nothing, all results should be written into algo_solution
}

void GreedyAlgo::verify(std::map<uint32_t, SegmentAlgoInfo>& segment_algo_infos,
                        std::map<uint32_t, uint16_t>& algo_solution, const uint32_t& cache_size) {
    assert(!segment_algo_infos.empty());
    assert(algo_solution.size() == segment_algo_infos.size());

    std::fstream f_algo;
    f_algo.open("/home/guoteng_20241228_135/WaLSM+/log/algo.log", std::ios::out | std::ios::app);
    f_algo << "[DEBUG] start to verify algo : " << std::endl;

    f_algo << "[DEBUG] segment_algo_infos size : " << segment_algo_infos.size() << std::endl;
    f_algo << "[DEBUG] algo_solution size : " << algo_solution.size() << std::endl;
    f_algo << "[DEBUG] cache size : " << cache_size << std::endl;
    assert(segment_algo_infos.size() == algo_solution.size());

    std::map<uint16_t, uint32_t> min_cnt_recorder, max_cnt_recorder;
    for (uint16_t i = 0; i <= MAX_UNITS_NUM; i++) {
        min_cnt_recorder[i] = 0xFFFFFFFFU; max_cnt_recorder[i] = 0;
    }                        

    // recheck that we already compute for all segments in segment_algo_infos
    auto infos_it = segment_algo_infos.begin();
    auto solution_it = algo_solution.begin();
    std::vector<uint32_t> segment_ids;
    uint32_t current_cache_size = 0;
    double ideal_cost = 0;
    while (infos_it != segment_algo_infos.end() && solution_it != algo_solution.end()) {
        assert(infos_it->first == solution_it->first);
        segment_ids.emplace_back(infos_it->first);
        current_cache_size += (infos_it->second.size_per_unit * solution_it->second);
        f_algo << "[DEBUG] segment " << infos_it->first << " , visit cnt : " 
                << infos_it->second.visit_cnt << " , size of each unit : " 
                << infos_it->second.size_per_unit << " , units num : " 
                << solution_it->second << std::endl;
        min_cnt_recorder[solution_it->second] = std::min(min_cnt_recorder[solution_it->second], infos_it->second.visit_cnt);
        max_cnt_recorder[solution_it->second] = std::max(max_cnt_recorder[solution_it->second], infos_it->second.visit_cnt);
        ideal_cost += StandardCostForDebug(infos_it->second.visit_cnt, solution_it->second);
        infos_it++; solution_it++;
    }
    assert(current_cache_size <= cache_size);
    f_algo << "[DEBUG] current cache size : " << current_cache_size << std::endl;
    for (uint16_t i = 0; i <= MAX_UNITS_NUM; i++) {
        f_algo << "[DEBUG] " << i << " units, min cnt: " << min_cnt_recorder[i] << ", max cnt: " << max_cnt_recorder[i] << std::endl;
    }
    f_algo << "[DEBUG] ideal I/O cost : " << ideal_cost << std::endl;

    // if visit cnt of segment i > visit cnt of segment j, then segment i should enable more units than segment j
    const size_t segment_size = segment_ids.size();
    for (size_t i=0; i<segment_size; i++) {
        for (size_t j=i+1; j<segment_size; j++) {
            uint32_t segment_id_i = segment_ids[i];
            uint32_t segment_id_j = segment_ids[j];
            if ((segment_algo_infos.find(segment_id_i)->second).visit_cnt >= (segment_algo_infos.find(segment_id_j)->second).visit_cnt) {
                assert(algo_solution[segment_id_i] >= algo_solution[segment_id_j]);
            } else {
                assert(algo_solution[segment_id_i] <= algo_solution[segment_id_j]);
            }
        }
    }

    f_algo << std::endl;
    f_algo.close();
};

} // namespace ROCKSDB_NAMESPACE