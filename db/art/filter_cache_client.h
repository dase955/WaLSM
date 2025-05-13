#pragma once

#include <iostream>
#include <cassert>
#include <task_thread_pool.hpp>
#include "macros.h"
#include "filter_cache.h" 
#include "table/block_based/parsed_full_filter_block.h"
#include "table/format.h"

namespace ROCKSDB_NAMESPACE {

class FilterCacheClient;
class FilterCacheManager;
class ParsedFullFilterBlock;

class FilterCacheClient {
private:
    task_thread_pool::task_thread_pool pool_{FILTER_CACHE_THREADS_NUM};
    FilterCacheManager filter_cache_manager_;
    // we need heat_buckets_ready_ to become true before filter_cache_ready_
    // In YCSB benchmark, we first load data (insert key-value pairs) then may try get operation
    // so we can guarantee that heat_buckets_ready_ become true before filter_cache_ready_
    bool heat_buckets_ready_; // the same as FilterCacheManager.heat_buckets_.is_ready()

    // background thread part of prepare_heat_buckets
    void do_prepare_heat_buckets(const std::string& key, std::unordered_map<uint32_t, std::vector<std::string>>* segment_info_recorder);

    // background thread part of retrain_or_keep_model
    void do_retrain_or_keep_model(std::vector<uint16_t>* features_nums_except_level_0, 
                                         const std::map<uint32_t, uint16_t>* level_recorder,
                                         const std::map<uint32_t, std::vector<RangeRatePair>>* segment_ranges_recorder,
                                         const std::map<uint32_t, uint32_t>* unit_size_recorder);

    // background thread part of check_key
    void do_hit_count_recorder(uint32_t segment_id);

    // background thread part of hit_heat_buckets
    void do_hit_heat_buckets(const std::string& key);

    // // background thread part of make_adjustment
    // void do_make_adjustment();

    // background thread part of batch_insert_segments
    void do_batch_insert_segments(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,
                                         std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder,
                                         std::map<uint32_t, uint16_t>& new_level_recorder, uint32_t level_0_base_count,
                                         std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder);

    // background thread part of batch_delete_segments
    void do_batch_delete_segments(std::vector<uint32_t>& merged_segment_ids);

    // background thread part of batch_move_segments
    void do_batch_move_segments(std::vector<uint32_t>& moved_segment_ids,
                                std::map<uint32_t, uint16_t>& old_level_recorder,
                                std::map<uint32_t, uint16_t>& move_level_recorder,
                                std::map<uint32_t, std::vector<RangeRatePair>>& move_segment_ranges_recorder);

    // background thread part of periods_work;
    void do_periods_work();

public:
    FilterCacheClient() {
        heat_buckets_ready_ = false;
    }

    std::vector<std::string>& range_seperators() {
        return filter_cache_manager_.range_seperators();
    }

    // corresponding to FilterCacheManager.make_heat_buckets_ready
    // segment_info_recorder is a map that recorder min key and max key of every segment, its value be like: [min key, max key]
    // because heat buckets is mainly used for model features, and model dont do any work on level 0 segments
    // so segment_info_recorder only need to record min key and max key of non level 0 segments
    // (we can modify micro SAMPLES_MAXCNT to fit in the YCSB load period, simply, SAMPLES_MAXCNT should be at least 50%-75% of load data num ???)
    // set SAMPLES_MAXCNT < YCSB load kv nums, to make sure that we can make heat_buckets ready in YCSB load period
    // if segment_info_recorder is empty, try default key ranges num and divide
    bool prepare_heat_buckets(const std::string& key, std::unordered_map<uint32_t, std::vector<std::string>>* const segment_info_recorder);

    // correspinding to FilterCacheManager work: monitor manager ready_work(), call manager make_clf_model_ready and train first model
    // lastly call update_cache_and_heap
    // features_nums recorders features num (key ranges num * 2 + 1) of every non level 0 segments
    // other arguments see FilterCacheManager make_clf_model_ready, try_retrain_model and update_cache_and_heap
    // please ensure that 3 recorders need to keep the same segments set, or error will occur in train func
    // you can use mutex in compaction and flushing to guarantee this
    // then when every long period end, try to retrain a new model or keep last model
    void retrain_or_keep_model(std::vector<uint16_t>* features_nums_except_level_0, 
                               const std::map<uint32_t, uint16_t>* level_recorder,
                               const std::map<uint32_t, std::vector<RangeRatePair>>* segment_ranges_recorder,
                               const std::map<uint32_t, uint32_t>* unit_size_recorder);

    // correespinding to FilterCacheManager work: check_key and hit_count_recorder
    // return FilterCacheManager.check_key() and leave hit_count_recorder to background
    std::vector<CachableEntry<ParsedFullFilterBlock>> get_filter_blocks(uint32_t segment_id);

    // every db get operation need one hit_heat_buckets
    void hit_heat_buckets(const std::string& key);

    // keep track of period count, update access counters and retrain classifier model
    void periods_work();

    // // heap based adjustment
    // void make_adjustment();

    // batch insert segments into filter cache manager, will also delete merged segments
    void batch_insert_segments(std::vector<uint32_t> merged_segment_ids, std::vector<uint32_t> new_segment_ids,
                               std::map<uint32_t, std::unordered_map<uint32_t, double>> inherit_infos_recorder,
                               std::map<uint32_t, uint16_t> new_level_recorder, uint32_t level_0_base_count,
                               std::map<uint32_t, std::vector<RangeRatePair>> segment_ranges_recorder);
    
    // In WaLSM+, we only support one column family, we just save cfd ptr here
    void update_cfd_ptr_if_needed(ColumnFamilyData* cfd); 

    // batch delete segments from filter cache manager
    void batch_delete_segments(std::vector<uint32_t> merged_segment_ids);

    // batch of moving segments to one level
    void batch_move_segments(std::vector<uint32_t> moved_segment_ids,
                             std::map<uint32_t, uint16_t> old_level_recorder,
                             std::map<uint32_t, uint16_t> move_level_recorder,
                             std::map<uint32_t, std::vector<RangeRatePair>> move_segment_ranges_recorder);
    
    
    void init_segment(uint32_t segment_id, const BlockBasedTable* table, const std::vector<BlockHandle>& block_handles);
};

}
