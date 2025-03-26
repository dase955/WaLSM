#pragma once

#include <iostream>
#include <fstream>
#include <mutex>
#include <cassert>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include "macros.h"
#include "greedy_algo.h"
#include "clf_model.h"
#include "heat_buckets.h"
#include "filter_cache_heap.h"
#include "filter_cache_item.h"

namespace ROCKSDB_NAMESPACE {

class FilterCache;
class FilterCacheManager;

// FilterCache main component is a STL Map, key -- segment id, value -- Structure of Filter Units （ called FilterCacheItem）
// its main job is auto enable/disable filter units for one segment, and check whether one key exists in enabled units
// its work is below:
// 1. enable / disable units for a batch of segments (one segment may not exist in FilterCache)
// 2. check whether one given key exists in one segment
// 3. check whether filter cache is approximately full
// 4. check whether ready to train first model
// 5. release FilterCacheItem of these merged (outdated) segments
class FilterCache {
private:
    std::map<uint32_t, FilterCacheItem> filter_cache_;
    uint32_t used_space_size_;
    uint32_t level_0_used_space_size_;
    uint32_t cache_size_; // max size of cache
    std::mutex filter_cache_mutex_;

public:
    FilterCache() { filter_cache_.clear(); cache_size_ = CACHE_SPACE_SIZE; used_space_size_ = 0; level_0_used_space_size_ = 0; }

    ~FilterCache() { /* do nothing */ }

    // other levels total cache size 
    uint32_t cache_size_except_level_0() { return cache_size_ * FULL_RATE - level_0_used_space_size_; }

    // check whether one given key exist in one segment
    bool check_key(const uint32_t& segment_id, const std::string& key);

    // enable / disable units for a batch of segments (one segment may not exist in FilterCache)
    // if enabled units num exceed given units num, it will disable units
    void enable_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder, const bool& is_forced,
                             std::set<uint32_t>& level_0_segment_ids, std::set<uint32_t>& failed_segment_ids);

    // the only difference from enable_for_segments is:
    // this func dont insert any filter units for segments that dont exist in cache, but enable_for_segments unc does
    void update_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder, const bool& is_forced,
                             std::set<uint32_t>& level_0_segment_ids, std::set<uint32_t>& failed_segment_ids);

    // check whether filter cache is approximately full
    // actually, we will leave (1-FULL_RATE) * cache_size_ space for emergency usage
    bool is_full();

    // check whether ready to train first model
    bool is_ready();

    // release filter units of merged segments
    void release_for_segments(std::vector<uint32_t>& segment_ids, std::set<uint32_t>& level_0_segment_ids);
};

// FilterCacheManager is combined of these components:
// HeatBuckets, ClfModel, FilterCacheHeapManager, GreedyAlgo and FilterCache
// its work is below:
// 1. input segment id and target key, check whether target key exist in this segment 
// 2. if one check done, add 1 to the get cnt of this segment. we need to maintain get cnts record in last long period and current long period
// 3. Inherit: convert get cnts of merged segments to get cnts of newly generated segments (get cnts in both last long period and current long period)
// 4. use existing counts of segments in last and current long period to estimate a approximate get cnt for one alive segment
// 5. record total get cnt and update short periods, reminded TRAIN_PERIODS short periods is a long period
// 6. if a short period end, update HeatBuckets
// 7. if a long period end, use greedy algorithm to solve filter units allocation problem and evaluate old model with this solution. if model doesnt work well, retrain it
// 8. if model still works well or model already retrained, we predict new ideal units num for current segments, release unnecessary filter units , add necessary filter unit and update FIlterCacheHeap
// 9. if old segments are merged, remove related filter units from FilterCache and FilterCacheHeap
// 10. if new segments are generated, we need to predict ideal units num for them, insert filter units (if cache have space left) , calcuate estimated count and update FIlterCacheHeap.
// 11. after one short period end, estimate current segments' approximate get cnts, then use these estimated get cnts to update FIlterCacheHeap 
// 12. before FilterCache becomes full for the first time, just set default units num for every new segments and insert filter units for segments
// 13. After FilterCache becomes full for the first time, start a background thread to monitor FitlerCacheHeap and use two-heaps adjustment to optimize FilterCache (this thread never ends)
class FilterCacheManager {
private:
    // TODO: mutex can be optimized or use a message queue or a thread pool to reduce time costed by mutex
    static FilterCache filter_cache_;
    static HeatBuckets heat_buckets_;
    static ClfModel clf_model_;
    static GreedyAlgo greedy_algo_;
    static FilterCacheHeapManager heap_manager_;
    static uint32_t get_cnt_; // record get cnt in current period, when exceeding PERIOD_COUNT, start next period
    static uint32_t period_cnt_; // record period cnt, if period_cnt_ - last_train_period_ >= TRAIN_PERIODS, start to evaluate or retrain ClfModel
    static uint32_t last_long_period_; // record last short period cnt of last long period
    static uint32_t last_short_period_; // helper var for update job when one short period ends
    static std::mutex update_mutex_; // guarantee counts records only updated once
    static bool train_signal_; // if true, try to retrain model. we call one background thread to monitor this flag and retrain
    static std::map<uint32_t, uint32_t> last_count_recorder_; // get cnt recorder of segments in last long period
    static std::map<uint32_t, uint32_t> current_count_recorder_; // get cnt recorder of segments in current long period
    static std::mutex count_mutex_; // guarentee last_count_recorder and current_count_recorder treated orderedly
    static bool is_ready_; // check whether ready to use adaptive filter assignment
public:
    FilterCacheManager() { get_cnt_ = 0; last_long_period_ = 0; last_short_period_ = 0; train_signal_ = false; }

    ~FilterCacheManager();

    // one background thread monitor this func, if return true, call try_retrain_model at once, wait for training end, and call update_cache_and_heap
    bool need_retrain() { return train_signal_; }

    // one background thread monitor this func, if return true, call make_clf_model_ready first, then call try_retrain_model at once and wait for training end. 
    // lastly call update_cache_and_heap. if all end, stop this thread, because if is_ready_ is true, is_ready_ will never change to false
    bool ready_work() { return is_ready_; }

    bool heat_buckets_ready() { return heat_buckets_.is_ready(); }

    // input segment id and target key, check whether target key exist in this segment 
    // return true when target key may exist (may cause false positive fault)
    // if there is no cache item for this segment, always return true
    // normal bloom filter units query, can we put hit_count_recorder outside this func? this will make get opt faster
    // will be called by a get operation, this will block get operation
    // remember to call hit_count_recorder in a background thread
    bool check_key(const uint32_t& segment_id, const std::string& key);

    // add 1 to get cnt of specified segment in current long period
    // will be called when calling check_key
    // remember to move this func to a single background thread aside check_key
    // because this func shouldn't block get operations
    void hit_count_recorder(const uint32_t& segment_id);

    // copy counts to last_count_recorder and reset counts of current_count_recorder
    void update_count_recorder();

    // inherit counts of merged segments to counts of new segments and remove counts of merged segments
    // inherit_infos_recorder: { {new segment 1: [{old segment 1: inherit rate 1}, {old segment 2: inherit rate 2}, ...]}, ...}
    void inherit_count_recorder(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids, const uint32_t& level_0_base_count,
                                std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder);

    // estimate approximate get cnts for every alive segment
    void estimate_counts_for_all(std::map<uint32_t, uint32_t>& approximate_counts_recorder);

    // noticed that at the beginning, heat buckets need to sample put keys to init itself before heat buckets start to work
    // segment_info_recorder is external variable that records every alive segments' min key and max key
    // it is like { segment 1: [min_key_1, max_key_1], segment 2: [min_key_2, max_key_2], ... }
    // return true when heat_buckets is ready, so no need to call this func again
    // remember to be called when receiving put opt. Normally, we can make heat_buckets_ ready before YCSB load end, so we can use it in YCSB testing phase
    // every put operation will call a background thread to call make_heat_buckets_ready
    // after this func return true, no need to call this func in put operation 
    // remember to move this func to a single background thread aside put operations
    bool make_heat_buckets_ready(const std::string& key, std::unordered_map<uint32_t, std::vector<std::string>>& segment_info_recorder);

    // clf_model_ need to determine feature nums before training
    // actually, YCSB will load data before testing
    // features_nums: [feature_num_1, feature_num_2, ...], it includes all feature_num of all non level 0 alive segments
    // feature_num_k is 2 * (number of key ranges intersecting with segment k) + 1
    // return true when clf_model_ set to ready successfully
    // we need to call make_clf_model_ready before we first call try_retrain_model
    // simply, if ready_work return true, we call make_clf_model_ready at once
    bool make_clf_model_ready(std::vector<uint16_t>& features_nums);

    // add 1 to get cnt of target key range for every get operation
    // update short periods if get cnt exceeds PERIOD_COUNT
    // every get opt will make add 1 to only one heat bucket counter 
    // also need to update count records if one long period end
    // also need to re-calcuate estimated count of current segments and update FilterCacheHeap if one short period end
    // we should use one background thread to call this func in every get operation
    void hit_heat_buckets(const std::string& key);

    // if one long period end, we need to check effectiveness of model. 
    // if model doesnt work well in current workload, we retrain this model
    // 1. use greedy algorithm to solve filter units allocation problem (receive ideal enabled units num for every current segments)
    // 2. write filter units nums and segment-related features to a csv file
    // 3. python lightgbm server maintain latest model. it will read the csv file and use I/O cost metric to check effectiveness of this model
    // 4. if effectiveness check not pass, retrain this model
    // reminded that if this new model training not end, lightgbm still use old model to predict ideal units num for segments
    // level_recorder: { segment 1: level_1, segment 2: level_2, ... }, level_k is the index of LSM-Tree level (0, 1, 2, ...)
    // range_heat_recorder: { segment 1: range_id_1, ...}, ... },
    // unit_size_recorder: { segment 1: unit_size_1, segment 2: unit_size_2, ... }
    // we assume for every segment, its ranges in range_heat_recorder value must be unique!!!
    // all 3 recorders need to maintain all current non level 0 segments info, and their keys size and keys set should be the same (their keys are segments' ids)
    // we ignore all level 0 segments !!! 3 recorders keys set should be the same ------ all alive segments' ids (except level 0)
    // because of the time cost of writing csv file, we need to do this func with a background thread
    // need real benchmark data to debug this func
    void try_retrain_model(std::map<uint32_t, uint16_t>& level_recorder,
                           std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder,
                           std::map<uint32_t, uint32_t>& unit_size_recorder);

    // after one long period end, we may retrain model. when try_retrain_model func end, we need to predict units num for every segments
    // then use these units num to update FilterCache, at last update units num limit in FilterCacheHeap
    // we ignore all level 0 segments !!!
    // level_recorder: { segment 1: level_1, segment 2: level_2, ... }, level_k is the index of LSM-Tree level (0, 1, 2, ...)
    // range_heat_recorder: { segment 1: range_id_1, ...}, ... },
    // noticed that we only pick up those segments that are in both level_recorder and segment_ranges_recorder
    // and update their filter units in filter cache and nodes in heap
    // so level_recorder keys set and segment_ranges_recorder keys set can be different
    // only be called after try_retrain_model (should be guaranteed)
    // we can guarantee this by putting try_retrain_model and update_cache_and_heap into only one background thread
    void update_cache_and_heap(std::map<uint32_t, uint16_t>& level_recorder,
                               std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder);

    // remove merged segments' filter units in the filter cache
    // also remove related items in FilterCacheHeap
    // segment_ids: [level_1_segment_1, level_0_segment_1, ...]
    // level_0_segment_ids: [level_0_segment_1, ...]
    // should be called by one background thread
    // this func will be called by insert_segments
    // you can also call this func alone after segments are merged (not suggested)
    void remove_segments(std::vector<uint32_t>& segment_ids, std::set<uint32_t>& level_0_segment_ids);

    // insert new segments into cache
    // all level 0 segments must enable all filter units
    // if is_ready_ is not true, set default filter units num (except level 0), insert into filter_cache_ and heaps
    // if is_ready_ is true, predict filter units num and insert necessary filter units as much as possible
    // then insert into heaps
    // for level 0 segments, we only need to insert all units into cache, dont insert any nodes into heaps
    // that means level 0 segments units num never be modified
    // merged_segment_ids can be null when new segments in level 0
    // level_0_base_count is the default value of level 0 segments' count in current recorder and last recorder
    // noticed that level 0 segments are from MemTable, we can compute total memtable get count and let the total count divided by new level 0 segments num
    // it equals to the avg count of new level 0 segments. we use this avg count to simply init level 0 segments' counts in current recorder and last recorder
    // merged_segment_ids: all merged segments' id
    // new_segment_ids: all new segments' id
    // inherit_infos_recorder: count inherit information (from merged segments to new segments), see inherit_count_recorder func
    // level_recorder: include merged segments and new segments, like { segment 1 : level_num_1, ... }
    // level_0_base_count: the initial count in last recorder and current recorder of new level 0 segments
    // segment_ranges_recorder: only include new segments, see update_cache_and_heap
    // level_recorder keys set and segment_ranges_recorder keys set can be different
    // but should ensure all new segments are in both level_recorder and segment_ranges_recorder
    // should be called by one background thread!
    // when old segments are merged into some new segments, call this func in one background thread
    void insert_segments(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,
                         std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder,
                         std::map<uint32_t, uint16_t>& level_recorder, const uint32_t& level_0_base_count,
                         std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder);

    // in func insert_segments above, we will also remove merged segments, this work well for normal compaction and flush
    // but we found that WaLSM also do delete compaction (only delete segments)
    // which not fit to func insert_segments, so we need a alone func delete_segments
    // this func only delete merged segments
    // we only need argument merged_segment_ids (all merged segments' ids)
    // and level_recorder which only include merged segments' level
    void delete_segments(std::vector<uint32_t>& merged_segment_ids, std::map<uint32_t, uint16_t>& level_recorder);

    // move segments to another level, used for trivial move compaction
    void move_segments(std::vector<uint32_t>& moved_segment_ids,
                       std::map<uint32_t, uint16_t>& old_level_recorder,
                       std::map<uint32_t, uint16_t>& move_level_recorder,
                       std::map<uint32_t, std::vector<RangeRatePair>>& move_segment_ranges_recorder);

    // make filter unit adjustment based on two heaps (benefit of enabling one unit & cost of disabling one unit)
    // simply, we disable one unit of one segment and enable one unit of another segment and guarantee cost < benefit
    // dont mind these two units size are not equal
    // in YCSB Benchmark, sizes of filter units are very close
    // return true when we successfully make one adjustment
    // return false when we cannot make one adjustment
    // one background should exec this func and never stop
    bool adjust_cache_and_heap();

    std::vector<std::string>& range_seperators() {
        return heat_buckets_.seperators();
    }
};

}