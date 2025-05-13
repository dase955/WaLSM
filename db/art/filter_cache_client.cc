#include "filter_cache_client.h"
#include <iostream>
#include <mutex>
#include <ostream>
#include "db/art/macros.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "db/art/global_filter_cache_context.h"

namespace ROCKSDB_NAMESPACE {

void FilterCacheClient::do_prepare_heat_buckets(const std::string& key, std::unordered_map<uint32_t, std::vector<std::string>>* segment_info_recorder) {
    filter_cache_manager_.make_heat_buckets_ready(key, *segment_info_recorder);
}

bool FilterCacheClient::prepare_heat_buckets(const std::string& key, std::unordered_map<uint32_t, std::vector<std::string>>* segment_info_recorder) {
    heat_buckets_ready_ = filter_cache_manager_.heat_buckets_ready();
    if (!heat_buckets_ready_) {
        // if heat_buckets_ready_ false
        assert(segment_info_recorder->size() == 0); // should always empty
        heat_buckets_ready_ = filter_cache_manager_.heat_buckets_ready();
        if (!heat_buckets_ready_) {
            // will leaks memory if pass ref
            // pool_.submit_detach([this, &key, segment_info_recorder]() {
                do_prepare_heat_buckets(key, segment_info_recorder);
            // });
            heat_buckets_ready_ = filter_cache_manager_.heat_buckets_ready();
        }
    }
    return heat_buckets_ready_;
}

void FilterCacheClient::do_retrain_or_keep_model(std::vector<uint16_t>* features_nums_except_level_0, 
                                                 const std::map<uint32_t, uint16_t>* level_recorder,
                                                 const std::map<uint32_t, std::vector<RangeRatePair>>* segment_ranges_recorder,
                                                 const std::map<uint32_t, uint32_t>* unit_size_recorder) {
    std::map<uint32_t, uint16_t> level_copy;
    std::map<uint32_t, std::vector<RangeRatePair>> segment_ranges_copy;
    std::map<uint32_t, uint32_t> unit_size_copy;
    bool clf_ready = false;
    bool clf_train = false; // if true, then clf model evaluated or retrained
    assert(READY_RATE <= FULL_RATE && READY_RATE >= 0);
    // if this func background monitor signal, how can it receive latest argument? input pointer!
    while (!filter_cache_manager_.heat_buckets_ready())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(filter_cache_manager_.heat_buckets_ready()); // must guarantee that heat buckets ready before we make filter cache manager ready
    std::cout << "[MODEL] heat buckets are ready." << std::endl;
    while (!filter_cache_manager_.ready_work())
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait for manager ready
    assert(filter_cache_manager_.ready_work());                                                
    std::cout << "[MODEL] filter cache is ready." << std::endl;
    
    // actually we will load data before we test, so we can ensure that heat buckets ready first
    std::cout << "[MODEL] model feature number: " << (*features_nums_except_level_0)[0] << std::endl;
    assert((*features_nums_except_level_0)[0] == MAX_FEATURES_NUM);
    clf_ready = filter_cache_manager_.make_clf_model_ready(*features_nums_except_level_0);
    assert(clf_ready);
    // lock and copy recorders
    global_filter_cache_recorders_mutex.lock();
    level_copy = *level_recorder; 
    segment_ranges_copy = *segment_ranges_recorder;
    unit_size_copy = *unit_size_recorder;
    global_filter_cache_recorders_mutex.unlock();
    assert(level_copy.size() == segment_ranges_copy.size());
    std::cout << "[MODEL] level recorder size (include level 0): " << level_copy.size() << std::endl;
    std::cout << "[MODEL] range recorder size (include level 0): " << segment_ranges_copy.size() << std::endl;
    // train first time, before that, there is no model left. 
    // Note: if it reach here when YCSB loading, we dont train model. we will train first model when one long period ends.
    clf_train = filter_cache_manager_.try_retrain_model(level_copy, segment_ranges_copy, unit_size_copy);
    if (UNLIKELY(clf_train)) {
        assert(false); // only train first model when YCSB load ends.
        std::cout << "[MODEL] we retrain a new model, thus we update filter cache and heaps" << std::endl;
        std::cout << "[MODEL] level recorder size (exclude level 0): " << level_copy.size() << std::endl;
        std::cout << "[MODEL] range recorder size (exclude level 0): " << segment_ranges_copy.size() << std::endl;
        filter_cache_manager_.update_cache_and_heap(level_copy, segment_ranges_copy); 
    }

    // retrain in long periods
    while (true) {
        bool adjusted = false;
        // in one long period
        while (!filter_cache_manager_.need_retrain()) {
            // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait for long period end
            adjusted = filter_cache_manager_.adjust_cache_and_heap();
            if (adjusted) std::cout << "[ADJUST] filter cache adjustment!" << std::endl;
        }
        assert(filter_cache_manager_.need_retrain());
        // lock and copy recorders
        global_filter_cache_recorders_mutex.lock();
        level_copy = *level_recorder; 
        segment_ranges_copy = *segment_ranges_recorder;
        unit_size_copy = *unit_size_recorder;
        global_filter_cache_recorders_mutex.unlock();
        assert(level_copy.size() == segment_ranges_copy.size());
        std::cout << "[MODEL] level recorder size (include level 0): " << level_copy.size() << std::endl;
        std::cout << "[MODEL] range recorder size (include level 0): " << segment_ranges_copy.size() << std::endl;
        // train first time, before that, there is no model left
        clf_train = filter_cache_manager_.try_retrain_model(level_copy, segment_ranges_copy, unit_size_copy);
        if (LIKELY(clf_train)) {
            std::cout << "[MODEL] we retrain a new model, thus we update filter cache and heaps" << std::endl;
            std::cout << "[MODEL] level recorder size (exclude level 0): " << level_copy.size() << std::endl;
            std::cout << "[MODEL] range recorder size (exclude level 0): " << segment_ranges_copy.size() << std::endl;
            filter_cache_manager_.update_cache_and_heap(level_copy, segment_ranges_copy);
        }
    }
    // this loop never end
}

void FilterCacheClient::retrain_or_keep_model(std::vector<uint16_t>* features_nums_except_level_0, 
                                              const std::map<uint32_t, uint16_t>* level_recorder,
                                              const std::map<uint32_t, std::vector<RangeRatePair>>* segment_ranges_recorder,
                                              const std::map<uint32_t, uint32_t>* unit_size_recorder) {
    pool_.submit_detach([this, features_nums_except_level_0, level_recorder, segment_ranges_recorder, unit_size_recorder]() {
        do_retrain_or_keep_model(features_nums_except_level_0, level_recorder, segment_ranges_recorder, unit_size_recorder);
    });
    // if first model training not end, python lgb_model server still return default units num
    // then retrain model when every long period end. if model still work well, keep this model instead
    // no need to return any value
}

// // TODO: make it a atomic operation rather than a mutex + threading
void FilterCacheClient::do_hit_count_recorder(uint32_t segment_id) {
    filter_cache_manager_.hit_count_recorder(segment_id);
}

std::vector<CachableEntry<ParsedFullFilterBlock>> FilterCacheClient::get_filter_blocks(uint32_t segment_id) {
    // pool_.submit_detach([this, segment_id]() {
    //     do_hit_count_recorder(segment_id);
    // });
    do_hit_count_recorder(segment_id);
    return filter_cache_manager_.get_filter_blocks(segment_id);
}

void FilterCacheClient::do_hit_heat_buckets(const std::string& key) {
    filter_cache_manager_.hit_heat_buckets(key);
}

void FilterCacheClient::hit_heat_buckets(const std::string& key) {
    // pool_.submit_detach([this, key]() {
    //     do_hit_heat_buckets(key);
    // });
    do_hit_heat_buckets(key);
}

void FilterCacheClient::do_periods_work() {
    while (true) {
        filter_cache_manager_.do_periods_work();
    }
}

void FilterCacheClient::periods_work() {
    pool_.submit_detach([this]() {
        do_periods_work();
    });
}

// void FilterCacheClient::do_make_adjustment() {
//     assert(false);
//     while (true) {
//         // never stop making heap adjustment
//         filter_cache_manager_.adjust_cache_and_heap();
//     }
// }

// void FilterCacheClient::make_adjustment() {
//     assert(false);
//     pool_.submit_detach([this]() {
//         do_make_adjustment();
//     });
// }

void FilterCacheClient::do_batch_insert_segments(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,
                                                 std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder,
                                                 std::map<uint32_t, uint16_t>& new_level_recorder, uint32_t level_0_base_count,
                                                 std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder) {
    filter_cache_manager_.insert_segments(merged_segment_ids, new_segment_ids, inherit_infos_recorder,
                                          new_level_recorder, level_0_base_count, segment_ranges_recorder);
}

void FilterCacheClient::batch_insert_segments(std::vector<uint32_t> merged_segment_ids, std::vector<uint32_t> new_segment_ids,
                                              std::map<uint32_t, std::unordered_map<uint32_t, double>> inherit_infos_recorder,
                                              std::map<uint32_t, uint16_t> new_level_recorder, uint32_t level_0_base_count,
                                              std::map<uint32_t, std::vector<RangeRatePair>> segment_ranges_recorder) {
    assert(new_segment_ids.size() == new_level_recorder.size());
    assert(new_segment_ids.size() == segment_ranges_recorder.size());
    assert(new_segment_ids.size() > 0);
    if (level_0_base_count == 0) {
        pool_.submit_detach([this, merged_segment_ids, new_segment_ids, inherit_infos_recorder, new_level_recorder, segment_ranges_recorder]() mutable {
            do_batch_insert_segments(merged_segment_ids, new_segment_ids, inherit_infos_recorder, new_level_recorder, INIT_LEVEL_0_COUNT, segment_ranges_recorder);
        });
    } else {
        pool_.submit_detach([this, merged_segment_ids, new_segment_ids, inherit_infos_recorder, new_level_recorder, level_0_base_count, segment_ranges_recorder]() mutable {
            do_batch_insert_segments(merged_segment_ids, new_segment_ids, inherit_infos_recorder, new_level_recorder, level_0_base_count, segment_ranges_recorder);
        });
    }
}

void FilterCacheClient::update_cfd_ptr_if_needed(ColumnFamilyData* cfd) {
    filter_cache_manager_.update_cfd(cfd);
}

void FilterCacheClient::do_batch_delete_segments(std::vector<uint32_t>& merged_segment_ids) {
    assert(false);
    exit(1);
    filter_cache_manager_.delete_segments(merged_segment_ids);
}

// disallowed in WaLSM+
void FilterCacheClient::batch_delete_segments(std::vector<uint32_t> merged_segment_ids) {
    assert(false);
    exit(1);
    pool_.submit_detach([this, merged_segment_ids]() mutable {
        do_batch_delete_segments(merged_segment_ids);
    });
}

void FilterCacheClient::do_batch_move_segments(std::vector<uint32_t>& moved_segment_ids,
                                               std::map<uint32_t, uint16_t>& old_level_recorder,
                                               std::map<uint32_t, uint16_t>& move_level_recorder,
                                               std::map<uint32_t, std::vector<RangeRatePair>>& move_segment_ranges_recorder) {
    assert(false);
    exit(1);
    filter_cache_manager_.move_segments(moved_segment_ids, old_level_recorder, move_level_recorder, move_segment_ranges_recorder);                                         
}

// disallowed in WaLSM+
void FilterCacheClient::batch_move_segments(std::vector<uint32_t> moved_segment_ids,
                                            std::map<uint32_t, uint16_t> old_level_recorder,
                                            std::map<uint32_t, uint16_t> move_level_recorder,
                                            std::map<uint32_t, std::vector<RangeRatePair>> move_segment_ranges_recorder) {
    assert(false);
    exit(1);
    assert(moved_segment_ids.size() == move_level_recorder.size());
    assert(moved_segment_ids.size() == move_segment_ranges_recorder.size());
    pool_.submit_detach([this, &moved_segment_ids, &old_level_recorder, &move_level_recorder, &move_segment_ranges_recorder]() {
        do_batch_move_segments(moved_segment_ids, old_level_recorder, move_level_recorder, move_segment_ranges_recorder);
    });
}

void FilterCacheClient::init_segment(uint32_t segment_id, const BlockBasedTable* table, const std::vector<BlockHandle>& block_handles) {
    assert(block_handles.size() > 0);
    filter_cache_manager_.init_segment(segment_id, table, block_handles);
}

}