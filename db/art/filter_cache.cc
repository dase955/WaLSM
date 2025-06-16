#include "filter_cache.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <chrono>
#include "table/block_based/parsed_full_filter_block.h"
#include "filter_cache_entry.h"
#include "port/likely.h"

namespace ROCKSDB_NAMESPACE {

std::vector<CachableEntry<ParsedFullFilterBlock>> FilterCache::get_filter_blocks(const uint32_t segment_id) {
    auto it = filter_cache_.find(segment_id);
    if (UNLIKELY(it == filter_cache_.end())) {
        // not in cache, that means we havent insert segment FilterCacheItem info into cache
        // actually, we start inserting after every segment becomes available
        // we return a empty vector here
        return {};
    }

    return it->second.get_filter_blocks();
}

void FilterCache::init_segment(uint32_t segment_id, const BlockBasedTable* table, const std::vector<BlockHandle>& block_handles) {
    // filter_cache_[segment_id] = FilterCacheEntry(segment_id, table, this, block_handles);

    if (LIKELY(table != nullptr && block_handles.size() == MAX_UNITS_NUM)) {
        filter_cache_.emplace(std::piecewise_construct, std::make_tuple(segment_id), std::make_tuple(segment_id, table, this, block_handles));
    }
}

void FilterCache::enable_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder, const bool& is_forced,
                                      std::set<uint32_t>& new_level_0_segment_ids, std::set<uint32_t>& failed_segment_ids) {
    failed_segment_ids.clear();
    filter_cache_mutex_.lock();
    // uint32_t enable_non_l0_count = 0, enable_l0_count = 0, fail_count = 0;
    // std::cout << "level 0 filter usage before enable: " << level_0_used_space_size_ << std::endl;
    // std::cout << "non level 0 filter usage before enable: " << used_space_size_ << std::endl; 
    for (auto it = segment_units_num_recorder.begin(); it != segment_units_num_recorder.end(); it ++) {
        const uint32_t segment_id = it->first;
        const uint16_t units_num = it->second;
        auto cache_it = filter_cache_.find(segment_id);
        bool is_level_0 = new_level_0_segment_ids.count(segment_id);
        if (cache_it != filter_cache_.end()) {
            // filter units cached
            const uint32_t old_size = (cache_it->second).approximate_size();
            assert(old_size >= 0); // should not cache it before
            if (is_forced || is_level_0 || !is_full()) {
                (cache_it->second).enable_units(units_num);
                if (is_level_0) {
                    level_0_used_space_size_ = level_0_used_space_size_ - old_size + (cache_it->second).approximate_size();
                    // enable_l0_count++;
                    // std::cout << "enable " << int((cache_it->second).approximate_size()) - int(old_size) 
                    //           << " bits for l0 segment " << segment_id  << ", units num: " << units_num << std::endl;
                }
                else {
                    used_space_size_ = used_space_size_ - old_size + (cache_it->second).approximate_size();
                    // enable_non_l0_count++;
                    // std::cout << "enable " << int((cache_it->second).approximate_size()) - int(old_size) 
                    //           << " bits for non l0 segment " << segment_id << ", units num: " << units_num << std::endl;
                }
            } else {
                failed_segment_ids.insert(segment_id);
                // fail_count++;
                assert(new_level_0_segment_ids.count(segment_id) == 0);
                // std::cout << "failed to enable filters for segment " << segment_id << std::endl;
            }
        } else {
            // already call FIlterCache::init_segment, 
            // so new segment already inserted into filter cache, but no filter units cached

            // filter units not cached
            // now cache it
            // if (is_forced || is_level_0 || !is_full()) {
            //     FilterCacheEntry cache_item(units_num);
            //     filter_cache_.insert(std::make_pair(segment_id, cache_item));
            //     used_space_size_ = used_space_size_ + cache_item.approximate_size();
            //     if (is_level_0) {
            //         level_0_used_space_size_ = level_0_used_space_size_ + cache_item.approximate_size();
            //     }
            // } else {
            //     failed_segment_ids.insert(segment_id);
            // }

            // all segments to be enabled must have been inited
            // std::cout << "filter handle not exist, segment id: " << segment_id << std::endl;
            // assert(false);
        }
    }
    // std::cout << "enable l0 count: " << enable_l0_count << ", enable non l0 count: " << enable_non_l0_count << ", fail count: " << fail_count << std::endl;
    // std::cout << "level 0 filter usage after enable: " << level_0_used_space_size_ << std::endl;
    // std::cout << "non level 0 filter usage after enable: " << used_space_size_ << std::endl; 
    // assert(enable_l0_count == new_level_0_segment_ids.size());
    // assert(enable_l0_count + enable_non_l0_count + fail_count == segment_units_num_recorder.size());
    filter_cache_mutex_.unlock();
}

void FilterCache::update_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder,
                                      std::set<uint32_t>& old_level_0_segment_ids, std::set<uint32_t>& failed_segment_ids) {
    assert(false); // only used in move_segment, but it is disallowed, so this func never used.
    exit(0);
    // because no new segments is generated, no reason to increase the usage of filter cache
    bool is_forced = true; 
    failed_segment_ids.clear();
    filter_cache_mutex_.lock();
    for (auto it = segment_units_num_recorder.begin(); it != segment_units_num_recorder.end(); it ++) {
        const uint32_t segment_id = it->first;
        const uint16_t units_num = it->second;
        auto cache_it = filter_cache_.find(segment_id);
        bool is_level_0 = old_level_0_segment_ids.count(segment_id);
        if (cache_it != filter_cache_.end()) {
            const uint32_t old_size = (cache_it->second).approximate_size();
            // filter units cached
            if (is_forced || is_level_0 || !is_full()) {
                (cache_it->second).enable_units(units_num);
                if (is_level_0) {
                    assert(old_size > 0); // should already cache filter for level 0.
                    level_0_used_space_size_ -= old_size;
                    used_space_size_ += (cache_it->second).approximate_size();
                } else {
                    used_space_size_ = used_space_size_ - old_size + (cache_it->second).approximate_size();
                }
            } else {
                // never reach this statement, because is_forced is always true
                assert(false);
                failed_segment_ids.insert(segment_id);
            }
        } else {
            // filter units not cached
            // do nothing!!!
            
            // all segments to be enabled must have been inited
            // std::cout << "error segment_id: " << segment_id << std::endl;
            // assert(false);
        }
    }
    filter_cache_mutex_.unlock();
}

bool FilterCache::is_full() {
    return double(used_space_size_) / double(cache_size_) >= FULL_RATE;
}

bool FilterCache::is_ready() {
    return double(used_space_size_) / double(cache_size_) >= READY_RATE;
}

void FilterCache::release_for_segments(std::vector<uint32_t>& segment_ids, std::set<uint32_t>& old_level_0_segment_ids) {
    std::sort(segment_ids.begin(), segment_ids.end());
    // delete key-value pair in filter_cache_
    filter_cache_mutex_.lock();
    auto it = filter_cache_.begin();
    size_t idx = 0;
    // uint32_t release_non_l0_count = 0, release_l0_count = 0;
    // std::cout << "level 0 filter usage before release: " << level_0_used_space_size_ << std::endl;
    // std::cout << "non level 0 filter usage before release: " << used_space_size_ << std::endl; 
    while (it != filter_cache_.end() && idx < segment_ids.size()) {
        if (it->first < segment_ids[idx]) {
            it ++;
        } else if (it->first > segment_ids[idx]) {
            idx ++;
        } else {
            if (old_level_0_segment_ids.count(it->first)) {
                level_0_used_space_size_ = level_0_used_space_size_ - (it->second).approximate_size();
                // release_l0_count++; 
                // std::cout << "free " << (it->second).approximate_size() << " bits of level 0 segment " << it->first << std::endl;
            } else {
                used_space_size_ = used_space_size_ - (it->second).approximate_size();
                // release_non_l0_count++;
                // std::cout << "free " << (it->second).approximate_size() << " bits of non level 0 segment " << it->first << std::endl;
            }
            it = filter_cache_.erase(it); idx++;
        }
    }
    // assert(release_non_l0_count + release_l0_count == segment_ids.size());
    // assert(release_l0_count == old_level_0_segment_ids.size());
    // std::cout << "release l0 count: " << release_l0_count << ", release non l0 count: " << release_non_l0_count << std::endl;
    // std::cout << "level 0 filter usage after release: " << level_0_used_space_size_ << std::endl;
    // std::cout << "non level 0 filter usage after release: " << used_space_size_ << std::endl; 
    filter_cache_mutex_.unlock();
}

bool FilterCacheManager::make_heat_buckets_ready(const std::string& key, 
                                                 std::unordered_map<uint32_t, std::vector<std::string>>& segment_info_recorder) {
    // heat_buckets not ready, still sample into pool
    if (!heat_buckets_.is_ready()) {
        std::vector<std::vector<std::string>> segments_infos;
        for (auto it = segment_info_recorder.begin(); it != segment_info_recorder.end(); it ++) {
            assert((it->second).size() == 2);
            segments_infos.emplace_back(it->second);
        }
        // segments_infos can be empty, then use default number of buckets
        heat_buckets_.sample(key, segments_infos);
    }
    return heat_buckets_.is_ready();
}

void FilterCacheManager::hit_heat_buckets(const std::string& key) {
    bool signal = false;
    if (LIKELY(heat_buckets_.is_ready())) {
        get_cnt_ += 1;
        heat_buckets_.hit(key, signal); // if one period end, return true signal
        if (signal) {
            period_mutex_.WriteLock();
            get_cnt_ = 0;
            period_cnt_ += 1;
            // std::cout << "get cnt updated, current period cnt: " << period_cnt_ << std::endl;
            period_mutex_.WriteUnlock();
        }
    }
}

void FilterCacheManager::do_periods_work() {
    bool need_retrain = false;

    // called by a background thread, never need to lock
    // update_mutex_.lock();
    if (period_cnt_ - last_long_period_ >= TRAIN_PERIODS) {
        // std::cout << "period_cnt_: " << period_cnt_ << std::endl;
        // std::cout << "last_long_period_: " << last_long_period_ << std::endl;
        last_long_period_ = period_cnt_;
        update_count_recorder();
        // debug_count_recorder();
        // std::map<uint32_t, uint32_t> recent_count_recorder;
        // std::vector<uint32_t> empty_needed_segment_ids;
        // estimate_recent_counts(recent_count_recorder, empty_needed_segment_ids);
        // assert(recent_count_recorder.size() > 0);
        // std::cout << "long period end, sync visit cnt." << std::endl;
        // heap_manager_.sync_visit_cnt(recent_count_recorder);
        train_signal_ = true;
        need_retrain = true;
    }
    if (period_cnt_ - last_short_period_ >= 1) {
        last_short_period_ = period_cnt_;
        // if already updated, do not update again
        if (!need_retrain) {
            // std::cout << "period_cnt_: " << period_cnt_ << std::endl;
            // std::cout << "last_short_period_: " << last_short_period_ << std::endl;
            // debug_count_recorder();
            // std::map<uint32_t, uint32_t> recent_count_recorder;
            // std::vector<uint32_t> empty_needed_segment_ids;
            // estimate_recent_counts(recent_count_recorder, empty_needed_segment_ids);
            // assert(recent_count_recorder.size() > 0);
            // std::cout << "short period end, sync visit cnt." << std::endl;
            // heap_manager_.sync_visit_cnt(recent_count_recorder);
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // update_mutex_.unlock();
}

bool FilterCacheManager::make_clf_model_ready(std::vector<uint16_t>& features_nums) {
    clf_model_.make_ready(features_nums);
    return clf_model_.is_ready();
}

std::vector<CachableEntry<ParsedFullFilterBlock>> FilterCacheManager::get_filter_blocks(uint32_t segment_id) {
    // move hit_count_recorder to a background thread
    // hit_count_recorder(segment_id); // one get opt will cause query to many segments.
    // so one get opt only call one hit_heat_buckets, but call many hit_count_recorder
    return filter_cache_.get_filter_blocks(segment_id);
}

void FilterCacheManager::init_segment(uint32_t segment_id, const BlockBasedTable* table, const std::vector<BlockHandle>& block_handles) {
    filter_cache_.init_segment(segment_id, table, block_handles);
}

void FilterCacheManager::hit_count_recorder(uint32_t segment_id) {
    count_mutex_.ReadLock();

    auto it = current_count_recorder_.find(segment_id);
    if (it == current_count_recorder_.end()) {
        // segment havent been visited, need to insert count
        // current_count_recorder_.insert(std::make_pair(segment_id, 1));
        // do nothing, wait for insertion
    } else {
        // segment have been visited, only update count
        it->second = it->second + 1;
    }

    count_mutex_.ReadUnlock();
}

void FilterCacheManager::update_count_recorder() {
    count_mutex_.WriteLock();

    last_count_recorder_.clear();
    // last_count_recorder_.insert(current_count_recorder_.begin(), current_count_recorder_.end());
    std::copy(current_count_recorder_.begin(), current_count_recorder_.end(), 
              std::inserter(last_count_recorder_, last_count_recorder_.begin()));
    assert(last_count_recorder_.size() == current_count_recorder_.size());
    for (auto it = current_count_recorder_.begin(); it != current_count_recorder_.end(); it++) {
        it->second = 0;
    }

    count_mutex_.WriteUnlock();
}

void FilterCacheManager::debug_count_recorder() {
    uint32_t get_cnt = 0;

    count_mutex_.ReadLock();

    std::cout << "last_count_recorder: " << std::endl;
    for (auto it = last_count_recorder_.begin(); it != last_count_recorder_.end(); it++) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
    std::cout << "current_count_recorder: " << std::endl;
    for (auto it = current_count_recorder_.begin(); it != current_count_recorder_.end(); it++) {
        get_cnt += it->second;
        std::cout << it->first << ": " << it->second << std::endl;
    }
    std::cout << "get_cnt: " << get_cnt << std::endl;

    count_mutex_.ReadUnlock();
}

void FilterCacheManager::inherit_count_recorder(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,  const uint32_t& level_0_base_count,
                                                std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder) {
    count_mutex_.WriteLock();

    // copy last count and current count of merged segments
    std::map<uint32_t, uint32_t> merged_last_count_recorder, merged_current_count_recorder; // cache merged segment count temporarily
    // std::cout << std::endl << std::endl;
    // std::cout << "merged segments id: ";
    for (uint32_t& merged_segment_id : merged_segment_ids) {
        merged_last_count_recorder.insert(std::make_pair(merged_segment_id, last_count_recorder_[merged_segment_id]));
        last_count_recorder_.erase(merged_segment_id);
        merged_current_count_recorder.insert(std::make_pair(merged_segment_id, current_count_recorder_[merged_segment_id]));
        current_count_recorder_.erase(merged_segment_id);

        // std::cout << merged_segment_id << " ";
        assert(last_count_recorder_.find(merged_segment_id) == last_count_recorder_.end());
        assert(current_count_recorder_.find(merged_segment_id) == current_count_recorder_.end());
    }
    // std::cout << std::endl;

    // std::cout << "merged segments size: " << merged_segment_ids.size() << std::endl;
    // std::cout << "new segments size: " << new_segment_ids.size() << std::endl;
    // std::cout << "inherit_infos_recorder size: " << inherit_infos_recorder.size() << std::endl;

    // init last count and current count of new segments based on inherit method (that not on Level 0)
    std::map<uint32_t, uint32_t> new_last_count_recorder, new_current_count_recorder;
    for (auto infos_it = inherit_infos_recorder.begin(); infos_it != inherit_infos_recorder.end(); infos_it ++) {
        double last_count = 0, current_count = 0;
        double weight_sum = 0;
        // std::cout << "child segment: " << infos_it->first << std::endl;
        std::unordered_map<uint32_t, double>& info = infos_it->second;
        for (auto info_it = info.begin(); info_it != info.end(); info_it ++) {
            last_count += INHERIT_REMAIN_FACTOR * (merged_last_count_recorder[info_it->first] * info_it->second);
            current_count += INHERIT_REMAIN_FACTOR * (merged_current_count_recorder[info_it->first] * info_it->second);
            weight_sum += info_it->second;
            // std::cout << "parent segment: " << info_it->first << " weight: " << info_it->second 
            //           << " last count: " << merged_last_count_recorder[info_it->first]
            //           << " current count: " << merged_current_count_recorder[info_it->first] << std::endl;
            assert(merged_last_count_recorder.find(info_it->first) != merged_last_count_recorder.end());
            assert(merged_current_count_recorder.find(info_it->first) != merged_current_count_recorder.end());
        }
        
        // std::cout << "temp last count: " << uint32_t(last_count) << " temp currrent count: " << uint32_t(current_count) << std::endl;
        // assert(weight_sum > 0.90);
        // weight sum should be 1.0, we multiple the inherited count by (1.0 / weight_sum)
        // assert(weight_sum > 0.98 && weight_sum < 1.02); // weight_sum approximately equals to 1.0
        last_count *= (1.0 / weight_sum); current_count *= (1.0 / weight_sum); // actually weight_sum always equals to 1.0
        // std::cout << "weight sum: " << weight_sum << " final last count: " << uint32_t(last_count) << " final currrent count: " << uint32_t(current_count) << std::endl;
        new_last_count_recorder.insert(std::make_pair(infos_it->first, uint32_t(last_count)));
        new_current_count_recorder.insert(std::make_pair(infos_it->first, uint32_t(current_count)));
    }

    assert(inherit_infos_recorder.size() == new_last_count_recorder.size());
    assert(inherit_infos_recorder.size() == new_current_count_recorder.size());
    assert(inherit_infos_recorder.size() <= new_segment_ids.size());

    // uint32_t last_insert_num = 0, current_insert_num = 0;
    // uint32_t last_update_num = 0, current_update_num = 0;
    // uint32_t last_check_num = 0, current_check_num = 0;

    // insert last count and current count of new segments
    for (uint32_t& new_segment_id : new_segment_ids) {
        // insert last count
        auto last_it = last_count_recorder_.find(new_segment_id);
        uint32_t new_last_count = level_0_base_count; // level 0 segments init
        // if true, this means new segment not on level 0, also means this segments are inherited from some segments
        if (new_last_count_recorder.count(new_segment_id) > 0) {
            new_last_count = new_last_count_recorder[new_segment_id];
            // last_check_num ++;
        }
        if (last_it != last_count_recorder_.end()) {
            last_it->second = last_it->second + new_last_count;
            // last_update_num ++;
        } else {
            last_count_recorder_.insert(std::make_pair(new_segment_id, new_last_count));
            // last_insert_num ++;
        }

        // insert current count
        auto current_it = current_count_recorder_.find(new_segment_id);
        uint32_t new_current_count = level_0_base_count; // level 0 segments init
        // if true, this means new segment not on level 0, also means this segments are inherited from some segments
        if (new_current_count_recorder.count(new_segment_id) > 0) {
            new_current_count = new_current_count_recorder[new_segment_id];
            // current_check_num ++;
        }
        if (current_it != current_count_recorder_.end()) {
            current_it->second = current_it->second + new_current_count;
            // current_update_num ++;
        } else {
            current_count_recorder_.insert(std::make_pair(new_segment_id, new_current_count));
            // current_insert_num ++;
        }

        assert(last_count_recorder_[new_segment_id] >= new_last_count);
        assert(current_count_recorder_[new_segment_id] >= new_current_count);
        // std::cout << "new segment id: " << new_segment_id << " last count: " << new_last_count << " current count: " << new_current_count << std::endl;
    }

    // assert(last_insert_num + last_update_num == new_segment_ids.size());
    // assert(current_insert_num + current_update_num == new_segment_ids.size());
    // assert(last_check_num == inherit_infos_recorder.size());
    // assert(current_check_num == inherit_infos_recorder.size());
    // std::cout << "last_insert_num: " << last_insert_num << " last_update_num: " << last_update_num << std::endl;
    // std::cout << "current_insert_num: " << current_insert_num << " current_update_num: " << current_update_num << std::endl;
    // std::cout << std::endl << std::endl;

    count_mutex_.WriteUnlock();
}

void FilterCacheManager::estimate_recent_counts(std::map<uint32_t, uint32_t>& approximate_counts_recorder, const std::vector<uint32_t>& needed_segment_ids) {
    const uint32_t long_period_total_count = TRAIN_PERIODS * PERIOD_COUNT;
    uint32_t current_long_period_count = PERIOD_COUNT * (period_cnt_ % TRAIN_PERIODS) + get_cnt_;
    double current_long_period_rate = std::min(double(current_long_period_count) / double(long_period_total_count), 1.0);

    if (needed_segment_ids.empty()) {
        count_mutex_.ReadLock();
        approximate_counts_recorder.clear();
        // approximate_counts_recorder.insert(current_count_recorder_.begin(), current_count_recorder_.end());
        std::copy(current_count_recorder_.begin(), current_count_recorder_.end(), 
                  std::inserter(approximate_counts_recorder, approximate_counts_recorder.begin()));
        assert(approximate_counts_recorder.size() == current_count_recorder_.size());
        auto approx_it = approximate_counts_recorder.begin();
        auto last_it = last_count_recorder_.begin();
        // std::cout << "estimate all segments' recent frequency." << std::endl;
        while (approx_it != approximate_counts_recorder.end() && last_it != last_count_recorder_.end()) {
            if (approx_it->first > last_it->first) {
                last_it ++;
            } else if(approx_it->first < last_it->first) {
                approx_it ++;
            } else {
                uint32_t recent_result = approx_it->second + uint32_t((1 - current_long_period_rate) * last_it->second);
                // if (current_long_period_rate > 0) {
                //     std::cout << "current rate: " << current_long_period_rate
                //             << ", current count: " << approx_it->second
                //             << ", last count: " << last_it->second
                //             << ", final recent count: " << recent_result << std::endl;
                // }
                approx_it->second = recent_result;
                assert(approximate_counts_recorder[approx_it->first] == recent_result);
                if (uint32_t((1 - current_long_period_rate) * last_it->second)  > 0) 
                    assert(current_count_recorder_[approx_it->first] != recent_result);
                approx_it ++;
                last_it ++;
            }
        }
        count_mutex_.ReadUnlock();
    } else {
        count_mutex_.ReadLock();
        approximate_counts_recorder.clear();
        for (uint32_t segment_id : needed_segment_ids) {
            approximate_counts_recorder.insert(std::make_pair(segment_id, current_count_recorder_[segment_id]));
        }
        assert(approximate_counts_recorder.size() == needed_segment_ids.size());
        auto approx_it = approximate_counts_recorder.begin();
        // std::cout << "estimate some segments' recent frequency." << std::endl;
        while (approx_it != approximate_counts_recorder.end()) {
            uint32_t recent_result = approx_it->second + 
                                     uint32_t((1 - current_long_period_rate) * last_count_recorder_[approx_it->first]);
            // if (current_long_period_rate > 0) {
            //     std::cout << "current rate: " << current_long_period_rate
            //               << ", current count: " << approx_it->second
            //               << ", last count: " << last_count_recorder_[approx_it->first]
            //               << ", final recent count: " << recent_result << std::endl;
            // }
            approx_it->second = recent_result;
            assert(approximate_counts_recorder[approx_it->first] == recent_result);
            if (uint32_t((1 - current_long_period_rate) * last_count_recorder_[approx_it->first]) > 0) 
                assert(current_count_recorder_[approx_it->first] != recent_result);
            approx_it ++;
        }
        count_mutex_.ReadUnlock();
    }
    // return nothing, already write result to approximate_counts_recorder
}


bool FilterCacheManager::try_retrain_model(std::map<uint32_t, uint16_t>& level_recorder,
                                           std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder,
                                           std::map<uint32_t, uint32_t>& unit_size_recorder) {
    // we should guarantee these 3 external recorder share the same keys set
    // we need to do this job outside FilterCacheManager
    assert(level_recorder.size() == segment_ranges_recorder.size());
    // assert(level_recorder.size() == unit_size_recorder.size());
    // should not train when loading, train_signal_ only true when starting YCSB run.
    if (train_signal_ == false) {
        return false;
    }

    // auto level_it_0 = level_recorder.begin();
    // while (level_it_0 != level_recorder.end()) {
    //     if (last_count_recorder_.find(level_it_0->first) == last_count_recorder_.end()) continue;
    //     uint32_t cnt = last_count_recorder_[level_it_0->first];
    //     std::cout << level_it_0->first << " : " << cnt << ", level: " << level_it_0->second << std::endl;
    //     level_it_0++;
    // }

    // recheck whether each segments include at least one key ranges.
    // auto ranges_it = segment_ranges_recorder.begin();
    // while(ranges_it != segment_ranges_recorder.end())
    // {
    //     // std::cout << "segment " << ranges_it->first 
    //     //           << " ranges num : " << (ranges_it->second).size() << std::endl;
    //     assert((ranges_it->second).size() > 0);

    //     double rate_sum = 0;
    //     for (RangeRatePair& pair : ranges_it->second) {
    //         rate_sum += pair.rate_in_segment;
    //     }
    //     assert(rate_sum <= 1.02 && rate_sum >= 0.98);

    //     ranges_it++;
    // }                                        

    // solve programming problem
    std::map<uint32_t, uint16_t> label_recorder;
    std::map<uint32_t, SegmentAlgoInfo> algo_infos;
    /*
    auto get_cnt_it = last_count_recorder_.begin();
    auto unit_size_it = unit_size_recorder.begin();
    while (unit_size_it != unit_size_recorder.end() && get_cnt_it != last_count_recorder_.end()) {
        if (unit_size_it->first > get_cnt_it->first) {
            get_cnt_it ++;
        } else if (unit_size_it->first < get_cnt_it->first) {
            unit_size_it ++;
        } else {
            algo_infos.insert(std::make_pair(unit_size_it->first, SegmentAlgoInfo(get_cnt_it->second, unit_size_it->second)));
            unit_size_it ++;
        }
    }
    greedy_algo_.solve(algo_infos, label_recorder, filter_cache_.cache_size_except_level_0());
    */
    assert(unit_size_recorder.size() == 0);

    std::map<uint32_t, uint32_t> last_count_recorder_copy;
    count_mutex_.ReadLock();
    last_count_recorder_copy = last_count_recorder_;
    count_mutex_.ReadUnlock();

    auto get_cnt_it = last_count_recorder_copy.begin();
    while (get_cnt_it != last_count_recorder_copy.end()) {
        // unit_size_recorder always empty, so we only use DEFAULT_UNIT_SIZE
        // exclude level 0 segments
        if (level_recorder[get_cnt_it->first] > 0) {
            algo_infos.insert(std::make_pair(get_cnt_it->first, SegmentAlgoInfo(get_cnt_it->second, DEFAULT_UNIT_SIZE)));
        }
        get_cnt_it ++;
    }
    assert(algo_infos.size() > 0);
    if (UNLIKELY(algo_infos.empty())) return false;
    std::cout << "[ALGO] algo_infos size: " << algo_infos.size() << std::endl;
    greedy_algo_.solve(algo_infos, label_recorder, filter_cache_.cache_size_except_level_0());
    std::cout << "[ALGO] stage 1: recorder size (exclude level 0): " << label_recorder.size() << std::endl;
    assert(algo_infos.size() == label_recorder.size());
    // // need to verify solutions
    // greedy_algo_.verify(algo_infos, label_recorder, filter_cache_.cache_size_except_level_0() / 256);
    std::map<uint16_t, uint32_t> min_cnt_recorder, max_cnt_recorder;
    for (uint16_t i = 0; i <= MAX_UNITS_NUM; i++) {
        min_cnt_recorder[i] = 0xFFFFFFFFU; max_cnt_recorder[i] = 0;
    }                        

    // recheck that we already compute for all segments in segment_algo_infos
    auto infos_it = algo_infos.begin();
    auto solution_it = label_recorder.begin();
    while (infos_it != algo_infos.end() && solution_it != label_recorder.end()) {
        assert(infos_it->first == solution_it->first);
        min_cnt_recorder[solution_it->second] = std::min(min_cnt_recorder[solution_it->second], infos_it->second.visit_cnt);
        max_cnt_recorder[solution_it->second] = std::max(max_cnt_recorder[solution_it->second], infos_it->second.visit_cnt);
        infos_it++; solution_it++;
    }
    adjust_manager_.UpdateCnt(min_cnt_recorder, max_cnt_recorder);

    // assert(level_recorder.size() == segment_ranges_recorder.size());
    // should make these two recorders share the same segment ids
    auto level_it_1 = level_recorder.begin();
    auto range_it_1 = segment_ranges_recorder.begin();
    while (level_it_1 != level_recorder.end()
           && range_it_1 != segment_ranges_recorder.end())
    {
        if (level_it_1->first < range_it_1->first) {
            level_it_1 = level_recorder.erase(level_it_1);
        } else if (level_it_1->first > range_it_1->first) {
            range_it_1 = segment_ranges_recorder.erase(range_it_1);
        } else {
            level_it_1++; range_it_1++;
        }
    }
    while (level_it_1 != level_recorder.end()) {
        level_it_1 = level_recorder.erase(level_it_1);
    }
    while (range_it_1 != segment_ranges_recorder.end()) {
        range_it_1 = segment_ranges_recorder.erase(range_it_1);
    }
    assert(level_recorder.size() == segment_ranges_recorder.size());

    // level_recorder and segment_ranges_recorder may include some merged segments, we need to ignore them
    auto old_level_it = level_recorder.begin();
    auto old_range_it = segment_ranges_recorder.begin();
    auto old_label_it = label_recorder.begin();
    while (old_level_it != level_recorder.end() && 
           old_range_it != segment_ranges_recorder.end() && 
           old_label_it != label_recorder.end()) {
        // std::cout << "debug : " << old_level_it->first << " : " << old_range_it->first << std::endl;
        assert(old_level_it->first == old_range_it->first);
        if (old_level_it->first < old_label_it->first) {
            old_level_it = level_recorder.erase(old_level_it);
            old_range_it = segment_ranges_recorder.erase(old_range_it);
        } else if (old_level_it->first > old_label_it->first) {
            old_label_it = label_recorder.erase(old_label_it);
        } else {
            old_level_it ++;
            old_range_it ++;
            old_label_it ++;
        }
    }
    // if some different elements remain in recorder's tail, we need to erase them
    while (old_level_it != level_recorder.end() && 
           old_range_it != segment_ranges_recorder.end()) {
        assert(old_level_it->first == old_range_it->first);
        old_level_it = level_recorder.erase(old_level_it);
        old_range_it = segment_ranges_recorder.erase(old_range_it);
    }
    while (old_label_it != label_recorder.end()) {
        old_label_it = label_recorder.erase(old_label_it);
    }

    // recheck whether these 3 recorder have same size
    assert(level_recorder.size() == segment_ranges_recorder.size());
    assert(level_recorder.size() == label_recorder.size());
    // auto check_level_it_2 = level_recorder.begin();
    // auto check_label_it_2 = label_recorder.begin();
    // while (check_level_it_2 != level_recorder.end()
    //        && check_label_it_2 != label_recorder.end())
    // {
    //     assert(check_level_it_2->first == check_label_it_2->first);
    //     check_level_it_2++; check_label_it_2++;
    // }
    std::cout << "[ALGO] stage 2: recorder size (exclude level 0): " << label_recorder.size() << std::endl;

    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::vector<std::vector<uint32_t>> datas;
    std::vector<uint16_t> labels;
    std::vector<uint32_t> get_cnts;

    std::cout << "[ALGO] stage 3: current count recorder size (include level 0): " << last_count_recorder_copy.size() << std::endl;
    // remember key range id starts with 0
    auto level_it = level_recorder.begin(); 
    auto range_it = segment_ranges_recorder.begin();
    auto count_it = last_count_recorder_copy.begin();
    auto label_it = label_recorder.begin();
    while (level_it != level_recorder.end() && range_it != segment_ranges_recorder.end() &&
           count_it != last_count_recorder_copy.end() && label_it != label_recorder.end()) {
        assert(level_it->first == range_it->first);
        assert(level_it->first == label_it->first);
        if (count_it->first < level_it->first) {
            count_it ++;
        } else if (count_it->first > level_it->first) {
            level_it ++;
            range_it ++;
            label_it ++;
        } else {
            // only train with non level 0 data
            if (LIKELY(level_it->second > 0)) {
                // add data row
                std::vector<uint32_t> data;
                std::vector<RangeHeatPair> heat_pairs;
                // double rate_sum = 0;
                for (RangeRatePair& pair : range_it->second) {
                    // rate_sum += pair.rate_in_segment;

                    RangeHeatPair heat_pair;
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    heat_pair.rate_in_segment = pair.rate_in_segment;
                    heat_pair.heat_value = buckets[pair.range_id].hotness_;
                    heat_pairs.emplace_back(heat_pair);
                }
                // assert(rate_sum >= 0.98 && rate_sum <= 1.02);
                assert(heat_pairs.size() == (range_it->second).size());

                std::sort(heat_pairs.begin(), heat_pairs.end(), RangeHeatPairGreatorComparor);
                for (size_t i = 0; i < heat_pairs.size() - 1; i ++) {
                    assert(heat_pairs[i].heat_value >= heat_pairs[i+1].heat_value);
                }
                
                data.emplace_back(level_it->second);
                for (RangeHeatPair& heat_pair : heat_pairs) {
                    data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * heat_pair.rate_in_segment));
                    data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * heat_pair.heat_value));
                }
                // std::cout << "[DEBUG] segment " << level_it->first << " data features num: " << data.size() << std::endl;
                assert(data.size() >= 3 && data.size() % 2 == 1);
                assert((range_it->second).size() * 2 + 1 == data.size());
                assert(data[0] > 0);
                datas.emplace_back(data);
                // add label row
                labels.emplace_back(label_it->second);
                assert(label_it->second <= MAX_UNITS_NUM);
                // add get cnt row
                get_cnts.emplace_back(count_it->second);
            }

            level_it ++;
            range_it ++;
            label_it ++;
        }
    }

    // check three vectors have same length
    assert(datas.size() == labels.size());
    assert(get_cnts.size() == labels.size());
    std::cout << "[ALGO] stage 3: training labels size (exclude level 0): " << labels.size() << std::endl;

    clf_model_.make_train(datas, labels, get_cnts);

    train_signal_ = false;

    return true;
}

void FilterCacheManager::update_cache_and_heap(std::map<uint32_t, uint16_t>& level_recorder,
                                               std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder) {
    std::vector<uint32_t> segment_ids;
    std::vector<std::vector<uint32_t>> datas;
    std::vector<uint16_t> preds;
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    std::map<uint32_t, uint16_t> current_units_num_limit_recorder;
    std::vector<Bucket> buckets = heat_buckets_.buckets();

    // check whether level 0 segments exist?
    // assert(level_recorder.size() == segment_ranges_recorder.size());
    // auto level_it_1 = level_recorder.begin();
    // auto range_it_1 = segment_ranges_recorder.begin();
    // while (level_it_1 != level_recorder.end()
    //        && range_it_1 != segment_ranges_recorder.end()) {
    //     assert(level_it_1->first == range_it_1->first);
    //     assert(level_it_1->second > 0);
    //     level_it_1++;
    //     range_it_1++;
    // }
    // check whether level 0 segments exist?
    // level_it_1 = level_recorder.begin();
    // while (level_it_1 != level_recorder.end()) {
    //     assert(level_it_1->second > 0);
    //     level_it_1++;
    // }

    // build data rows into datas
    auto level_it = level_recorder.begin();
    auto range_it = segment_ranges_recorder.begin();
    assert(level_recorder.size() == segment_ranges_recorder.size());
    while (level_it != level_recorder.end() && range_it != segment_ranges_recorder.end()) {
        if (level_it->first < range_it->first) {
            level_it ++;
        } else if (level_it->first > range_it->first) {
            range_it ++;
        } else {
            assert(level_it->first == range_it->first);
            assert(level_it->second > 0);
            if (LIKELY(level_it->second > 0)) {
                segment_ids.emplace_back(level_it->first);
                // add data row
                std::vector<uint32_t> data;
                std::vector<RangeHeatPair> heat_pairs;
                // double rate_sum = 0;
                for (RangeRatePair& pair : range_it->second) {
                    // rate_sum += pair.rate_in_segment;

                    RangeHeatPair heat_pair;
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    heat_pair.rate_in_segment = pair.rate_in_segment;
                    heat_pair.heat_value = buckets[pair.range_id].hotness_;
                    heat_pairs.emplace_back(heat_pair);
                }
                // assert(rate_sum >= 0.98 && rate_sum <= 1.02);
                assert(heat_pairs.size() == (range_it->second).size());

                std::sort(heat_pairs.begin(), heat_pairs.end(), RangeHeatPairGreatorComparor);
                for (size_t i = 0; i < heat_pairs.size() - 1; i ++) {
                    assert(heat_pairs[i].heat_value >= heat_pairs[i+1].heat_value);
                }
                
                data.emplace_back(level_it->second);
                for (RangeHeatPair& heat_pair : heat_pairs) {
                    data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * heat_pair.rate_in_segment));
                    data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * heat_pair.heat_value));
                }
                // std::cout << "[DEBUG] segment " << level_it->first << " data features num: " << data.size() << std::endl;
                assert(data.size() >= 3 && data.size() % 2 == 1);
                assert((range_it->second).size() * 2 + 1 == data.size());
                assert(data[0] > 0);
                datas.emplace_back(data);
            }

            level_it ++;
            range_it ++;
        }
    }

    // use datas to make prediction
    clf_model_.make_predict(datas, preds);
    assert(segment_ids.size() == preds.size());
    size_t idx = 0;
    // std::cout << std::endl << "sync units num limit" << std::endl;
    while (idx < segment_ids.size() && idx < preds.size()) {
        segment_units_num_recorder.insert(std::make_pair(segment_ids[idx], preds[idx]));
        current_units_num_limit_recorder.insert(std::make_pair(segment_ids[idx], preds[idx]));
        // std::cout << "segment id: " << segment_ids[idx] << ", units limit: " << preds[idx] << std::endl;
        idx = idx + 1;
    }
    assert(segment_units_num_recorder.size() == current_units_num_limit_recorder.size());
    assert(segment_ids.size() == segment_units_num_recorder.size());

    // update filter cache helper heaps
    // heap_manager_.sync_units_num_limit(current_units_num_limit_recorder);
    adjust_manager_.UpdateLimit(current_units_num_limit_recorder);

    // update filter cache
    std::set<uint32_t> empty_level_0_segment_ids; // no level 0 segment in heaps and model data, dont worry
    std::set<uint32_t> empty_failed_segment_ids; 
    filter_cache_.enable_for_segments(segment_units_num_recorder, true, empty_level_0_segment_ids, empty_failed_segment_ids);
    assert(empty_failed_segment_ids.empty());

}

bool FilterCacheManager::adjust_cache_and_heap() {
    if ((!is_ready_) || !filter_cache_.is_full()) {
        return false;
    }
    FilterCacheModifyResult result;
    /*
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
    */
    // bool can_adjust = heap_manager_.try_modify(result);
    // if (can_adjust) {
    //     std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    //     std::set<uint32_t> empty_level_0_segment_ids; // no level 0 segment in heaps, dont worry
    //     std::set<uint32_t> empty_failed_segment_ids; // force to update segments' filter units group, so dont worry for cache space
    //     segment_units_num_recorder.insert(std::make_pair(result.enable_segment_id, result.enable_segment_next_units_num));
    //     segment_units_num_recorder.insert(std::make_pair(result.disable_segment_id, result.disable_segment_next_units_num));
    //     filter_cache_.enable_for_segments(segment_units_num_recorder, true, empty_level_0_segment_ids, empty_failed_segment_ids);
    //     assert(empty_failed_segment_ids.empty());
    // } 
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));

    bool can_adjust = adjust_manager_.adjust(result);
    if (can_adjust) {
        std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
        std::set<uint32_t> empty_level_0_segment_ids; // no level 0 segment in heaps, dont worry
        std::set<uint32_t> empty_failed_segment_ids; // force to update segments' filter units group, so dont worry for cache space
        segment_units_num_recorder.insert(std::make_pair(result.enable_segment_id, result.enable_units_num));
        segment_units_num_recorder.insert(std::make_pair(result.disable_segment_id, result.disable_units_num));
        filter_cache_.enable_for_segments(segment_units_num_recorder, true, empty_level_0_segment_ids, empty_failed_segment_ids);
        assert(empty_failed_segment_ids.empty());
    }
    return can_adjust;
}

void FilterCacheManager::insert_segments(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,
                                         std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder,
                                         std::map<uint32_t, uint16_t>& new_level_recorder, const uint32_t& level_0_base_count,
                                         std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder) {
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    // std::map<uint32_t, uint32_t> approximate_counts_recorder;
    std::set<uint32_t> failed_segment_ids;
    // std::vector<FilterCacheHeapItem> new_segment_items;
    std::set<uint32_t> old_level_0_segment_ids, new_level_0_segment_ids;
    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::sort(merged_segment_ids.begin(), merged_segment_ids.end());
    std::sort(new_segment_ids.begin(), new_segment_ids.end());

    assert(new_segment_ids.size() == new_level_recorder.size());
    assert(new_segment_ids.size() == segment_ranges_recorder.size());
    assert(new_segment_ids.size() >= inherit_infos_recorder.size());
    // uint32_t new_l0_count = 0, new_non_l0_count = 0;
    // size_t cached_l0_count = cached_level_0_segment_ids_.size();
    assert(DEFAULT_UNITS_NUM <= MAX_UNITS_NUM && DEFAULT_UNITS_NUM >= MIN_UNITS_NUM);
    for (auto& item : new_level_recorder) {
        auto segment_id = item.first;
        auto level = item.second;
        if (level == 0) {
            new_level_0_segment_ids.insert(segment_id);
            cached_level_0_segment_ids_.insert(segment_id); // update current cached level 0 segments
            segment_units_num_recorder.insert(std::make_pair(segment_id, MAX_UNITS_NUM));
            // new_l0_count++;
        } else {
            segment_units_num_recorder.insert(std::make_pair(segment_id, DEFAULT_UNITS_NUM));
            // new_non_l0_count++;
        }
    }
    // cached_l0_count += new_l0_count;
    // assert(new_l0_count + new_non_l0_count == new_segment_ids.size());
    assert(segment_units_num_recorder.size() == new_segment_ids.size());
    
    // // print new segment ids
    // std::cout << std::endl;
    // std::cout << "new level-0 segment id: ";
    // for (uint32_t segment_id : new_level_0_segment_ids) {
    //     std::cout << segment_id << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "new non level-0 segment id: ";
    // for (uint32_t segment_id : new_segment_ids) {
    //     if (new_level_0_segment_ids.count(segment_id) == 0) {
    //         std::cout << segment_id << " ";
    //     }
    // }
    // std::cout << std::endl;

    // collect old segments id on level 0
    // uint32_t old_l0_count = 0, old_non_l0_count = 0;
    // std::cout << "merged segment ids: " << std::endl;
    for (uint32_t& merged_segment_id : merged_segment_ids) {
        if (cached_level_0_segment_ids_.count(merged_segment_id) > 0) {
            old_level_0_segment_ids.insert(merged_segment_id);
            cached_level_0_segment_ids_.erase(merged_segment_id);
            // old_l0_count++;
        } else { 
            // old_non_l0_count++; 
            // std::cout << merged_segment_id << " ";
        }
    }
    // std::cout << std::endl;

    // cached_l0_count -= old_l0_count;
    // assert(cached_l0_count == cached_level_0_segment_ids_.size());
    // assert(old_l0_count + old_non_l0_count == merged_segment_ids.size());

    // // print merged segment ids
    // std::cout << "merged level-0 segment id: ";
    // for (uint32_t segment_id : old_level_0_segment_ids) {
    //     std::cout << segment_id << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "merged non level-0 segment id: ";
    // for (uint32_t segment_id : merged_segment_ids) {
    //     if (old_level_0_segment_ids.count(segment_id) == 0) {
    //         std::cout << segment_id << " ";
    //     }
    // }
    // std::cout << std::endl;

    if (!is_ready_) {
        // if is_ready_ is false, no need to enable two-heaps adjustment, remember to update is_ready_ in the end
        // remove merged segments' units in filter cache and nodes in filter heaps
        std::vector<uint32_t> merged_segment_ids_except_l0;
        for (uint32_t &segment_id : merged_segment_ids) {
            if (old_level_0_segment_ids.count(segment_id) == 0) {
                merged_segment_ids_except_l0.emplace_back(segment_id);
                adjust_manager_.DelSegment(segment_id);
            }
        }
        // heap_manager_.batch_delete(merged_segment_ids_except_l0);
        // std::cout << merged_segment_ids_except_l0.size() << " " << merged_segment_ids.size() << std::endl;
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // inherit merged segments' counts to new segments' counts
        // ensure that new segments that are not in inherit_infos_recorder keys set are only level 0 segments
        // this function will remove moved segments from last_count_recorder_ and current_count_recorder_
        inherit_count_recorder(merged_segment_ids, new_segment_ids, level_0_base_count, inherit_infos_recorder);

        // std::vector<uint32_t> needed_segment_ids;
        // for (uint32_t segment_id : new_segment_ids) {
        //     needed_segment_ids.emplace_back(segment_id);
        // }
        // estimate_recent_counts(approximate_counts_recorder, needed_segment_ids);
        // assert(approximate_counts_recorder.size() > 0);

        // insert units into filter cache
        filter_cache_.enable_for_segments(segment_units_num_recorder, false, new_level_0_segment_ids, failed_segment_ids);
        
        // insert nodes into filter heaps
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id) > 0) {
                // no need to insert level 0 segment nodes into heap
                continue;
            } else if (failed_segment_ids.count(new_segment_id) > 0) {
                // failed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                // new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                //                                                    0, 0, units_num));
                adjust_manager_.AddSegment(new_segment_id, 0, units_num);
            } else {
                // succeed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                // new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                //                                                    units_num, 0, units_num));
                adjust_manager_.AddSegment(new_segment_id, units_num, units_num);
            }
        }
        // assert(new_segment_items.size() + new_level_0_segment_ids.size() == new_segment_ids.size());
        // heap_manager_.batch_upsert(new_segment_items);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // is_ready_ is true, then we will not update is_ready_, that means is_ready_ will be always true
        // remove merged segments' units in filter cache and nodes in filter heaps
        std::vector<uint32_t> merged_segment_ids_except_l0;
        for (uint32_t &segment_id : merged_segment_ids) {
            if (old_level_0_segment_ids.count(segment_id) == 0) {
                merged_segment_ids_except_l0.emplace_back(segment_id);
                adjust_manager_.DelSegment(segment_id);
            }
        }
        // heap_manager_.batch_delete(merged_segment_ids_except_l0);
        // std::cout << merged_segment_ids_except_l0.size() << " " << merged_segment_ids.size() << std::endl;
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // inherit merged segments' counts to new segments' counts
        // ensure that new segments that are not in inherit_infos_recorder keys set are only level 0 segments
        // this function will remove moved segments from last_count_recorder_ and current_count_recorder_
        inherit_count_recorder(merged_segment_ids, new_segment_ids, level_0_base_count, inherit_infos_recorder);

        // std::vector<uint32_t> needed_segment_ids;
        // for (uint32_t segment_id : new_segment_ids) {
        //     needed_segment_ids.emplace_back(segment_id);
        // }
        // estimate_recent_counts(approximate_counts_recorder, needed_segment_ids);
        // assert(approximate_counts_recorder.size() > 0);

        // predict units num for new non level 0 segments and update segment_units_num_recorder
        std::vector<std::vector<uint32_t>> pred_datas;
        std::vector<uint32_t> pred_segment_ids;
        std::vector<uint16_t> pred_results;
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id) > 0) {
                // no need to predict for level 0 segments
                continue;
            } else {
                pred_segment_ids.emplace_back(new_segment_id);

                auto range_it = segment_ranges_recorder.find(new_segment_id);
                assert(range_it != segment_ranges_recorder.end());
                std::vector<uint32_t> pred_data;
                std::vector<RangeHeatPair> heat_pairs;
                // double rate_sum = 0;
                for (RangeRatePair& pair : range_it->second) {
                    // rate_sum += pair.rate_in_segment;

                    RangeHeatPair heat_pair;
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    heat_pair.rate_in_segment = pair.rate_in_segment;
                    heat_pair.heat_value = buckets[pair.range_id].hotness_;
                    heat_pairs.emplace_back(heat_pair);
                }
                // assert(rate_sum >= 0.98 && rate_sum <= 1.02);
                assert(heat_pairs.size() == (range_it->second).size());

                std::sort(heat_pairs.begin(), heat_pairs.end(), RangeHeatPairGreatorComparor);
                for (size_t i = 0; i < heat_pairs.size() - 1; i ++) {
                    assert(heat_pairs[i].heat_value >= heat_pairs[i+1].heat_value);
                }
                
                pred_data.emplace_back(new_level_recorder[new_segment_id]);
                for (RangeHeatPair& heat_pair : heat_pairs) {
                    pred_data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * heat_pair.rate_in_segment));
                    pred_data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * heat_pair.heat_value));
                }
                // std::cout << "[DEBUG] segment " << level_it->first << " data features num: " << data.size() << std::endl;
                assert(pred_data.size() >= 3 && pred_data.size() % 2 == 1);
                assert((range_it->second).size() * 2 + 1 == pred_data.size());
                assert(pred_data[0] > 0);
                pred_datas.emplace_back(pred_data);
            }
        }
        assert(pred_datas.size() == pred_segment_ids.size());
        clf_model_.make_predict(pred_datas, pred_results);
        assert(pred_datas.size() == pred_results.size());
        size_t pred_idx = 0;
        while (pred_idx < pred_segment_ids.size() && pred_idx < pred_results.size()) {
            segment_units_num_recorder[pred_segment_ids[pred_idx]] = pred_results[pred_idx];
            assert(new_level_0_segment_ids.count(pred_segment_ids[pred_idx]) == 0);
            assert(pred_results[pred_idx] >= MIN_UNITS_NUM && pred_results[pred_idx] <= MAX_UNITS_NUM);
            pred_idx = pred_idx + 1;
        }
        // std::cout << "insert predict " << pred_results.size() << " segments" << std::endl;
        assert(pred_results.size() + new_level_0_segment_ids.size() == new_segment_ids.size());

        // insert units into filter cache
        filter_cache_.enable_for_segments(segment_units_num_recorder, false, new_level_0_segment_ids, failed_segment_ids);

        // insert nodes into filter heaps
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id) > 0) {
                // no need to insert level 0 segment nodes into heap
                continue;
            } else if (failed_segment_ids.count(new_segment_id) > 0) {
                // failed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                // new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                //                                                    0, 0, units_num));
                adjust_manager_.AddSegment(new_segment_id, 0, units_num);
            } else {
                // succeed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                // new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                //                                                    units_num, 0, units_num));
                adjust_manager_.AddSegment(new_segment_id, units_num, units_num);
            }
        }
        // assert(new_segment_items.size() + new_level_0_segment_ids.size() == new_segment_ids.size());
        // heap_manager_.batch_upsert(new_segment_items);
    }
}

void FilterCacheManager::delete_segments(std::vector<uint32_t>& merged_segment_ids) {
    assert(false);
    exit(1);
    std::set<uint32_t> old_level_0_segment_ids;

    // collect old segments id on level 0
    for (uint32_t& merged_segment_id : merged_segment_ids) {
        if (cached_level_0_segment_ids_.count(merged_segment_id)) {
            old_level_0_segment_ids.insert(merged_segment_id);
            cached_level_0_segment_ids_.erase(merged_segment_id);
        }
        // remove merged segments' count
        last_count_recorder_.erase(merged_segment_id);
        current_count_recorder_.erase(merged_segment_id);
    }

    if (!is_ready_) {
        // if is_ready_ is false, no need to enable two-heaps adjustment, remember to update is_ready_ in the end
        // remove merged segments' units in filter cache and nodes in filter heaps
        // heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // is_ready_ is true, then we will not update is_ready_, that means is_ready_ will be always true
        // remove merged segments' units in filter cache and nodes in filter heaps
        // heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);
    }
}

void FilterCacheManager::move_segments(std::vector<uint32_t>& moved_segment_ids,
                                       std::map<uint32_t, uint16_t>& old_level_recorder,
                                       std::map<uint32_t, uint16_t>& move_level_recorder,
                                       std::map<uint32_t, std::vector<RangeRatePair>>& move_segment_ranges_recorder) {
    assert(false);
    exit(1);
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    // std::map<uint32_t, uint32_t> approximate_counts_recorder;
    // std::vector<FilterCacheHeapItem> new_segment_items;
    std::set<uint32_t> old_level_0_segment_ids;
    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::sort(moved_segment_ids.begin(), moved_segment_ids.end());

    // pick up merged or new level 0 segments, but this type of compaction must not move to level 0,
    // so we may only move level 0 to level below
    assert(moved_segment_ids.size() == old_level_recorder.size());
    assert(moved_segment_ids.size() == move_level_recorder.size());
    assert(moved_segment_ids.size() == move_segment_ranges_recorder.size());
    auto level_it = old_level_recorder.begin();
    size_t moved_idx = 0;
    while (level_it != old_level_recorder.end()) {
        assert(moved_idx < moved_segment_ids.size() && level_it->first == moved_segment_ids[moved_idx]);
        if (moved_idx < moved_segment_ids.size() && level_it->first == moved_segment_ids[moved_idx]) {
            segment_units_num_recorder.insert(std::make_pair(level_it->first, DEFAULT_UNITS_NUM));
            // actually, we cannot move segments to level 0 in trivial move compaction (only flushing do this).
            moved_idx ++;
        }
        level_it ++;
    }

    // collect old segments id on level 0
    for (uint32_t moved_segment_id : moved_segment_ids) {
        if (cached_level_0_segment_ids_.count(moved_segment_id)) {
            old_level_0_segment_ids.insert(moved_segment_id);
            cached_level_0_segment_ids_.erase(moved_segment_id);
        }
    }

    if (!is_ready_) {
        // firstly, delete moved segments
        // heap_manager_.batch_delete(moved_segment_ids);

        // inherit these segments' count
        // for (uint32_t& segment_id : moved_segment_ids) {
        //     auto last_it = last_count_recorder_.find(segment_id);
        //     auto current_it = current_count_recorder_.find(segment_id);
        //     if (last_it != last_count_recorder_.end()) {
        //         last_it->second = INHERIT_REMAIN_FACTOR * (last_it->second);
        //     }
        //     if (current_it != current_count_recorder_.end()) {
        //         current_it->second = INHERIT_REMAIN_FACTOR * (current_it->second);
        //     }
        // }
        // std::vector<uint32_t> needed_segment_ids;
        // for (uint32_t segment_id : moved_segment_ids) {
        //     needed_segment_ids.emplace_back(segment_id);
        // }
        // estimate_recent_counts(approximate_counts_recorder, needed_segment_ids);
        // assert(approximate_counts_recorder.size() > 0);

        // modify units into filter cache
        std::set<uint32_t> empty_failed_segment_ids;
        filter_cache_.update_for_segments(segment_units_num_recorder, old_level_0_segment_ids, empty_failed_segment_ids);
        
        // insert nodes into filter heaps
        // for (uint32_t& segment_id : moved_segment_ids) {
        //     assert(move_level_recorder[segment_id] > 0);
        //     uint16_t units_num = segment_units_num_recorder[segment_id];
        //     new_segment_items.emplace_back(FilterCacheHeapItem(segment_id, approximate_counts_recorder[segment_id],
        //                                                        units_num, 0, units_num));
        // }
        // heap_manager_.batch_upsert(new_segment_items);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // firstly, delete moved segments
        // heap_manager_.batch_delete(moved_segment_ids);

        // inherit these segments' count
        // for (uint32_t& segment_id : moved_segment_ids) {
        //     auto last_it = last_count_recorder_.find(segment_id);
        //     auto current_it = current_count_recorder_.find(segment_id);
        //     if (last_it != last_count_recorder_.end()) {
        //         last_it->second = INHERIT_REMAIN_FACTOR * (last_it->second);
        //     }
        //     if (current_it != current_count_recorder_.end()) {
        //         current_it->second = INHERIT_REMAIN_FACTOR * (current_it->second);
        //     }
        // }
        // std::vector<uint32_t> needed_segment_ids;
        // for (uint32_t segment_id : moved_segment_ids) {
        //     needed_segment_ids.emplace_back(segment_id);
        // }
        // estimate_recent_counts(approximate_counts_recorder, needed_segment_ids);
        // assert(approximate_counts_recorder.size() > 0);

        // predict units num for new non level 0 segments and update segment_units_num_recorder
        std::vector<std::vector<uint32_t>> pred_datas;
        std::vector<uint32_t> pred_segment_ids;
        std::vector<uint16_t> pred_results;
        for (uint32_t moved_segment_id : moved_segment_ids) {
            assert(move_level_recorder[moved_segment_id] > 0);
            pred_segment_ids.emplace_back(moved_segment_id);

            auto range_it = move_segment_ranges_recorder.find(moved_segment_id);
            assert(range_it != move_segment_ranges_recorder.end());
            std::vector<uint32_t> pred_data;
            std::vector<RangeHeatPair> heat_pairs;
            double rate_sum = 0;
            for (RangeRatePair& pair : range_it->second) {
                rate_sum += pair.rate_in_segment;

                RangeHeatPair heat_pair;
                assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                heat_pair.rate_in_segment = pair.rate_in_segment;
                heat_pair.heat_value = buckets[pair.range_id].hotness_;
                heat_pairs.emplace_back(heat_pair);
            }
            assert(rate_sum >= 0.98 && rate_sum <= 1.02);
            assert(heat_pairs.size() == (range_it->second).size());

            std::sort(heat_pairs.begin(), heat_pairs.end(), RangeHeatPairGreatorComparor);
            for (size_t i = 0; i < heat_pairs.size() - 1; i ++) {
                assert(heat_pairs[i].heat_value >= heat_pairs[i+1].heat_value);
            }
                
            pred_data.emplace_back(move_level_recorder[moved_segment_id]);
            for (RangeHeatPair& heat_pair : heat_pairs) {
                pred_data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * heat_pair.rate_in_segment));
                pred_data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * heat_pair.heat_value));
            }
            // std::cout << "[DEBUG] segment " << level_it->first << " data features num: " << data.size() << std::endl;
            assert(pred_data.size() >= 3 && pred_data.size() % 2 == 1);
            assert((range_it->second).size() * 2 + 1 == pred_data.size());
            assert(pred_data[0] > 0);
            pred_datas.emplace_back(pred_data);
        }
        assert(pred_datas.size() == pred_segment_ids.size());
        clf_model_.make_predict(pred_datas, pred_results);
        assert(pred_datas.size() == pred_results.size());
        size_t pred_idx = 0;
        while (pred_idx < pred_segment_ids.size() && pred_idx < pred_results.size()) {
            segment_units_num_recorder[pred_segment_ids[pred_idx]] = pred_results[pred_idx];
            pred_idx = pred_idx + 1;
        }
        assert(pred_results.size() == moved_segment_ids.size());

        // modify units into filter cache
        std::set<uint32_t> empty_failed_segment_ids;
        filter_cache_.update_for_segments(segment_units_num_recorder, old_level_0_segment_ids, empty_failed_segment_ids);

        // insert nodes into filter heaps
        // for (uint32_t segment_id : moved_segment_ids) {
        //     assert(move_level_recorder[segment_id] > 0);
        //     uint16_t units_num = segment_units_num_recorder[segment_id];
        //     new_segment_items.emplace_back(FilterCacheHeapItem(segment_id, approximate_counts_recorder[segment_id],
        //                                                        units_num, 0, units_num));
            
        // }
        // heap_manager_.batch_upsert(new_segment_items);
    }
}

    const char* FilterCache::Name() const { return "FilterCache"; }

    // overrides rocksdb::Cache but no nothing
    Status FilterCache::Insert(const Slice& key, void* value, size_t charge,
                            void (*deleter)(const Slice& key, void* value),
                            Handle** handle,
                            Priority priority) {
                                assert(false);
                                return Status::OK();
                            }

    // overrides rocksdb::Cache but no nothing
    Cache::Handle* FilterCache::Lookup(const Slice& key, Statistics* stats) {
        assert(false);
        return nullptr;
    }

    // overrides rocksdb::Cache but no nothing
    bool FilterCache::Ref(Handle* handle) { return false; }

    // used by CachableEntry
    bool FilterCache::Release(Cache::Handle* handle, bool force_erase) { return false; }

    // overrides rocksdb::Cache but no nothing
    void* FilterCache::Value(Cache::Handle* handle) { assert(false); return nullptr; }

    // overrides rocksdb::Cache but no nothing
    void FilterCache::Erase(const Slice& key) { assert(false); }
    // overrides rocksdb::Cache but no nothing
    uint64_t FilterCache::NewId() { assert(false); return 0; }

    // overrides rocksdb::Cache but no nothing
    void FilterCache::SetCapacity(size_t capacity) { assert(false); }

    // overrides rocksdb::Cache but no nothing
    void FilterCache::SetStrictCapacityLimit(bool strict_capacity_limit) { assert(false);}

    // overrides rocksdb::Cache but no nothing
    bool FilterCache::HasStrictCapacityLimit() const { assert(false); return false; }

    // overrides rocksdb::Cache but no nothing
    size_t FilterCache::GetCapacity() const { assert(false);  return 0; }

    // overrides rocksdb::Cache but no nothing
    size_t FilterCache::GetUsage() const { assert(false); return 0; }

    // overrides rocksdb::Cache but no nothing
    size_t FilterCache::GetUsage(Handle* handle) const { assert(false); return 0; }

    // overrides rocksdb::Cache but no nothing
    size_t FilterCache::GetPinnedUsage() const { assert(false); return 0; }

    // overrides rocksdb::Cache but no nothing
    size_t FilterCache::GetCharge(Handle* handle) const { assert(false); return 0; }

    // overrides rocksdb::Cache but no nothing
    void FilterCache::ApplyToAllCacheEntries(void (*callback)(void*, size_t),
                                        bool thread_safe) { assert(false); }

    // overrides rocksdb::Cache but no nothing
    void FilterCache::EraseUnRefEntries() { assert(false); }
}