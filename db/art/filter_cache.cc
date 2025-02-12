#include "filter_cache.h"
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

FilterCache FilterCacheManager::filter_cache_;
HeatBuckets FilterCacheManager::heat_buckets_;
ClfModel FilterCacheManager::clf_model_;
GreedyAlgo FilterCacheManager::greedy_algo_;
FilterCacheHeapManager FilterCacheManager::heap_manager_;
uint32_t FilterCacheManager::get_cnt_;
uint32_t FilterCacheManager::period_cnt_;
uint32_t FilterCacheManager::last_long_period_;
uint32_t FilterCacheManager::last_short_period_;
std::mutex FilterCacheManager::update_mutex_;
bool FilterCacheManager::train_signal_;
std::map<uint32_t, uint32_t> FilterCacheManager::last_count_recorder_; 
std::map<uint32_t, uint32_t> FilterCacheManager::current_count_recorder_; 
std::mutex FilterCacheManager::count_mutex_;
bool FilterCacheManager::is_ready_;

bool FilterCache::check_key(const uint32_t& segment_id, const std::string& key) {
    auto it = filter_cache_.find(segment_id);
    if (it == filter_cache_.end()) {
        // not in cache, that means we havent insert segment FilterCacheItem info into cache
        // actually, we start inserting after every segment becomes available
        return true;
    } else {
        return (it->second).check_key(key);
    }
}

void FilterCache::enable_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder, const bool& is_forced,
                                      std::set<uint32_t>& level_0_segment_ids, std::set<uint32_t>& failed_segment_ids) {
    failed_segment_ids.clear();
    filter_cache_mutex_.lock();
    for (auto it = segment_units_num_recorder.begin(); it != segment_units_num_recorder.end(); it ++) {
        const uint32_t segment_id = it->first;
        const uint16_t units_num = it->second;
        auto cache_it = filter_cache_.find(segment_id);
        bool is_level_0 = level_0_segment_ids.count(segment_id);
        if (cache_it != filter_cache_.end()) {
            // filter units cached
            if (is_forced || is_level_0 || !is_full()) {
                const uint32_t old_size = (cache_it->second).approximate_size();
                (cache_it->second).enable_units(units_num);
                used_space_size_ = used_space_size_ - old_size + (cache_it->second).approximate_size();
                if (is_level_0) {
                    level_0_used_space_size_ = level_0_used_space_size_ - old_size + (cache_it->second).approximate_size();
                }
            } else {
                failed_segment_ids.insert(segment_id);
            }
        } else {
            // filter units not cached
            // now cache it
            if (is_forced || is_level_0 || !is_full()) {
                FilterCacheItem cache_item(units_num);
                filter_cache_.insert(std::make_pair(segment_id, cache_item));
                used_space_size_ = used_space_size_ + cache_item.approximate_size();
                if (is_level_0) {
                    level_0_used_space_size_ = level_0_used_space_size_ + cache_item.approximate_size();
                }
            } else {
                failed_segment_ids.insert(segment_id);
            }
        }
    }
    filter_cache_mutex_.unlock();
}

void FilterCache::update_for_segments(std::unordered_map<uint32_t, uint16_t>& segment_units_num_recorder, const bool& is_forced,
                                      std::set<uint32_t>& level_0_segment_ids, std::set<uint32_t>& failed_segment_ids) {
    filter_cache_mutex_.lock();
    for (auto it = segment_units_num_recorder.begin(); it != segment_units_num_recorder.end(); it ++) {
        const uint32_t segment_id = it->first;
        const uint16_t units_num = it->second;
        auto cache_it = filter_cache_.find(segment_id);
        bool is_level_0 = level_0_segment_ids.count(segment_id);
        if (cache_it != filter_cache_.end()) {
            // filter units cached
            if (is_forced || is_level_0 || !is_full()) {
                const uint32_t old_size = (cache_it->second).approximate_size();
                (cache_it->second).enable_units(units_num);
                used_space_size_ = used_space_size_ - old_size + (cache_it->second).approximate_size();
                if (is_level_0) {
                    level_0_used_space_size_ = level_0_used_space_size_ - old_size + (cache_it->second).approximate_size();
                }
            } else {
                failed_segment_ids.insert(segment_id);
            }
        } else {
            // filter units not cached
            // do nothing!!!
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

void FilterCache::release_for_segments(std::vector<uint32_t>& segment_ids, std::set<uint32_t>& level_0_segment_ids) {
    std::sort(segment_ids.begin(), segment_ids.end());
    // delete key-value pair in filter_cache_
    filter_cache_mutex_.lock();
    auto it = filter_cache_.begin();
    size_t idx = 0;
    while (it != filter_cache_.end() && idx < segment_ids.size()) {
        if (it->first < segment_ids[idx]) {
            it ++;
        } else if (it->first > segment_ids[idx]) {
            idx ++;
        } else {
            used_space_size_ = used_space_size_ - (it->second).approximate_size();
            if (level_0_segment_ids.count(it->first)) {
                level_0_used_space_size_ = level_0_used_space_size_ - (it->second).approximate_size();
            }
            it = filter_cache_.erase(it);
        }
    }
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
    if (heat_buckets_.is_ready()) {
        get_cnt_ += 1;
        if (get_cnt_ >= PERIOD_COUNT) {
            heat_buckets_.hit(key, true);
            get_cnt_ = 0;
            period_cnt_ += 1;
        } else {
            heat_buckets_.hit(key, false);
        }
    }
    if (period_cnt_ - last_long_period_ >= TRAIN_PERIODS) {
        update_mutex_.lock();

        if (period_cnt_ - last_long_period_ >= TRAIN_PERIODS) {
            last_long_period_ = period_cnt_;
            update_count_recorder();
            train_signal_ = true;
        }

        update_mutex_.unlock();
    }
    if (period_cnt_ - last_short_period_ >= 1) {
        update_mutex_.lock();

        if (period_cnt_ - last_short_period_ >= 1) {
            last_short_period_ = period_cnt_;
            std::map<uint32_t, uint32_t> estimate_count_recorder;
            estimate_counts_for_all(estimate_count_recorder);
            heap_manager_.sync_visit_cnt(estimate_count_recorder);
        }

        update_mutex_.unlock();
    }
}

bool FilterCacheManager::make_clf_model_ready(std::vector<uint16_t>& features_nums) {
    clf_model_.make_ready(features_nums);
    return clf_model_.is_ready();
}

bool FilterCacheManager::check_key(const uint32_t& segment_id, const std::string& key) {
    // move hit_count_recorder to a background thread
    // hit_count_recorder(segment_id); // one get opt will cause query to many segments.
    // so one get opt only call one hit_heat_buckets, but call many hit_count_recorder
    return filter_cache_.check_key(segment_id, key);
}

void FilterCacheManager::hit_count_recorder(const uint32_t& segment_id) {
    count_mutex_.lock();

    auto it = current_count_recorder_.find(segment_id);
    if (it == current_count_recorder_.end()) {
        // segment havent been visited, need to insert count
        current_count_recorder_.insert(std::make_pair(segment_id, 1));
    } else {
        // segment have been visited, only update count
        it->second = it->second + 1;
    }

    count_mutex_.unlock();
}

void FilterCacheManager::update_count_recorder() {
    count_mutex_.lock();

    last_count_recorder_.clear();
    last_count_recorder_.insert(current_count_recorder_.begin(), current_count_recorder_.end());
    for (auto it = current_count_recorder_.begin(); it != current_count_recorder_.end(); it++) {
        it->second = 0;
    }

    count_mutex_.unlock();
}

void FilterCacheManager::inherit_count_recorder(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,  const uint32_t& level_0_base_count,
                                                std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder) {
    count_mutex_.lock();

    std::map<uint32_t, uint32_t> merged_last_count_recorder, merged_current_count_recorder; // cache merged segment count temporarily
    for (uint32_t& merged_segment_id : merged_segment_ids) {
        merged_last_count_recorder.insert(std::make_pair(merged_segment_id, last_count_recorder_[merged_segment_id]));
        last_count_recorder_.erase(merged_segment_id);
        merged_current_count_recorder.insert(std::make_pair(merged_segment_id, current_count_recorder_[merged_segment_id]));
        current_count_recorder_.erase(merged_segment_id);
    }

    std::map<uint32_t, uint32_t> new_last_count_recorder, new_current_count_recorder;
    for (auto infos_it = inherit_infos_recorder.begin(); infos_it != inherit_infos_recorder.end(); infos_it ++) {
        double last_count = 0, current_count = 0;
        std::unordered_map<uint32_t, double>& info = infos_it->second;
        for (auto info_it = info.begin(); info_it != info.end(); info_it ++) {
            last_count = last_count + INHERIT_REMAIN_FACTOR * (merged_last_count_recorder[info_it->first] * info_it->second);
            current_count = current_count + INHERIT_REMAIN_FACTOR * (merged_current_count_recorder[info_it->first] * info_it->second);
        }
        new_last_count_recorder.insert(std::make_pair(infos_it->first, uint32_t(last_count)));
        new_current_count_recorder.insert(std::make_pair(infos_it->first, uint32_t(current_count)));
    }

    for (uint32_t& new_segment_id : new_segment_ids) {
        auto last_it = last_count_recorder_.find(new_segment_id);
        uint32_t new_last_count = level_0_base_count; // level 0 segments init
        if (new_last_count_recorder.count(new_segment_id) > 0) {
            new_last_count = new_last_count_recorder[new_segment_id];
        }
        if (last_it != last_count_recorder_.end()) {
            last_it->second = last_it->second + new_last_count;
        } else {
            last_count_recorder_.insert(std::make_pair(new_segment_id, new_last_count));
        }

        auto current_it = current_count_recorder_.find(new_segment_id);
        uint32_t new_current_count = level_0_base_count; // level 0 segments init
        if (new_current_count_recorder.count(new_segment_id) > 0) {
            new_current_count = new_current_count_recorder[new_segment_id];
        }
        if (current_it != current_count_recorder_.end()) {
            current_it->second = current_it->second + new_current_count;
        } else {
            current_count_recorder_.insert(std::make_pair(new_segment_id, new_current_count));
        }
    }

    count_mutex_.unlock();
}

void FilterCacheManager::estimate_counts_for_all(std::map<uint32_t, uint32_t>& approximate_counts_recorder) {
    const uint32_t long_period_total_count = TRAIN_PERIODS * PERIOD_COUNT;
    uint32_t current_long_period_count = PERIOD_COUNT * (period_cnt_ % TRAIN_PERIODS) + get_cnt_;
    double current_long_period_rate = std::min(double(current_long_period_count) / double(long_period_total_count), 1.0);

    approximate_counts_recorder.clear();
    approximate_counts_recorder.insert(current_count_recorder_.begin(), current_count_recorder_.end());
    auto approx_it = approximate_counts_recorder.begin();
    auto last_it = last_count_recorder_.begin();
    while (approx_it != approximate_counts_recorder.end() && last_it != last_count_recorder_.end()) {
        if (approx_it->first > last_it->first) {
            last_it ++;
        } else if(approx_it->first < last_it->first) {
            approx_it ++;
        } else {
            approx_it->second = approx_it->second + uint32_t((1 - current_long_period_rate) * last_it->second);
            approx_it ++;
        }
    }

    // return nothing, already write result to approximate_counts_recorder
}


void FilterCacheManager::try_retrain_model(std::map<uint32_t, uint16_t>& level_recorder,
                                           std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder,
                                           std::map<uint32_t, uint32_t>& unit_size_recorder) {
    // we should guarantee these 3 external recorder share the same keys set
    // we need to do this job outside FilterCacheManager
    assert(level_recorder.size() == segment_ranges_recorder.size());
    // assert(level_recorder.size() == unit_size_recorder.size());
    if (train_signal_ == false) {
        return;
    }

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
    auto get_cnt_it = last_count_recorder_.begin();
    while (get_cnt_it != last_count_recorder_.end()) {
        // unit_size_recorder always empty, so we only use DEFAULT_UNIT_SIZE
        algo_infos.insert(std::make_pair(get_cnt_it->first, SegmentAlgoInfo(get_cnt_it->second, DEFAULT_UNIT_SIZE)));
        get_cnt_it ++;
    }
    greedy_algo_.solve(algo_infos, label_recorder, filter_cache_.cache_size_except_level_0());

    // programming problem may include some merged segments, we need to ignore them
    auto old_level_it = level_recorder.begin();
    auto old_range_it = segment_ranges_recorder.begin();
    auto old_label_it = label_recorder.begin();
    while (old_level_it != level_recorder.end() && 
           old_range_it != segment_ranges_recorder.end() && 
           old_label_it != label_recorder.end()) {
        assert(old_level_it->first == old_range_it->first);
        if (old_level_it->first < old_label_it->first) {
            old_level_it = level_recorder.erase(old_label_it);
            old_range_it = segment_ranges_recorder.erase(old_range_it);
        } else if (old_level_it->first > old_label_it->first) {
            old_label_it = label_recorder.erase(old_label_it);
        } else {
            old_level_it ++;
            old_range_it ++;
            old_label_it ++;
        }
    }
    while (old_level_it != level_recorder.end() && 
           old_range_it != segment_ranges_recorder.end()) {
        assert(old_level_it->first == old_range_it->first);
        old_level_it = level_recorder.erase(old_label_it);
        old_range_it = segment_ranges_recorder.erase(old_range_it);
    }
    while (old_label_it != label_recorder.end()) {
        old_label_it = label_recorder.erase(old_label_it);
    }

    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::vector<std::vector<uint32_t>> datas;
    std::vector<uint16_t> labels;
    std::vector<uint32_t> get_cnts;

    auto level_it = level_recorder.begin(); // key range id start with 0
    auto range_it = segment_ranges_recorder.begin();
    auto count_it = last_count_recorder_.begin();
    auto label_it = label_recorder.begin();
    while (level_it != level_recorder.end() && range_it != segment_ranges_recorder.end() &&
           count_it != last_count_recorder_.end() && label_it != label_recorder.end()) {
        assert(level_it->first == range_it->first);
        assert(level_it->first == label_it->first);
        if (count_it->first < level_it->first) {
            count_it ++;
        } else if (count_it->first > level_it->first) {
            level_it ++;
            range_it ++;
            label_it ++;
        } else {
            if (level_it->second > 0) {
                // add data row
                std::vector<uint32_t> data;
                std::sort((range_it->second).begin(), (range_it->second).end(), RangeRatePairGreatorComparor);
                data.emplace_back(level_it->second);
                for (RangeRatePair& pair : range_it->second) {
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * pair.rate_in_segment));
                    data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * buckets[pair.range_id].hotness_));
                }
                datas.emplace_back(data);
                // add label row
                labels.emplace_back(label_it->second);
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

    clf_model_.make_train(datas, labels, get_cnts);

    train_signal_ = false;
}

void FilterCacheManager::update_cache_and_heap(std::map<uint32_t, uint16_t>& level_recorder,
                                               std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder) {
    assert(level_recorder.size() == segment_ranges_recorder.size());
    std::vector<uint32_t> segment_ids;
    std::vector<std::vector<uint32_t>> datas;
    std::vector<uint16_t> preds;
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    std::map<uint32_t, uint16_t> current_units_num_limit_recorder;
    std::vector<Bucket> buckets = heat_buckets_.buckets();

    // build data rows into datas
    auto level_it = level_recorder.begin();
    auto range_it = segment_ranges_recorder.begin();
    while (level_it != level_recorder.end() && range_it != segment_ranges_recorder.end()) {
        if (level_it->first < range_it->first) {
            level_it ++;
        } else if (level_it->first > range_it->first) {
            range_it ++;
        } else {
            assert(level_it->first == range_it->first);

            if (level_it->second > 0) {
                // add data row
                std::vector<uint32_t> data;
                std::sort((range_it->second).begin(), (range_it->second).end(), RangeRatePairGreatorComparor);
                data.emplace_back(level_it->second);
                for (RangeRatePair& pair : range_it->second) {
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * pair.rate_in_segment));
                    data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * buckets[pair.range_id].hotness_));
                }
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
    while (idx < segment_ids.size() && idx < preds.size()) {
        segment_units_num_recorder.insert(std::make_pair(segment_ids[idx], preds[idx]));
        current_units_num_limit_recorder.insert(std::make_pair(segment_ids[idx], preds[idx]));
        idx = idx + 1;
    }

    // update filter cache helper heaps
    heap_manager_.sync_units_num_limit(current_units_num_limit_recorder);

    // update filter cache
    std::set<uint32_t> empty_level_0_segment_ids; // no level 0 segment in heaps and model data, dont worry
    std::set<uint32_t> empty_failed_segment_ids; 
    filter_cache_.update_for_segments(segment_units_num_recorder, true, empty_level_0_segment_ids, empty_failed_segment_ids);
}

void FilterCacheManager::remove_segments(std::vector<uint32_t>& segment_ids, std::set<uint32_t>& level_0_segment_ids) {
    // update filter cache helper heaps
    heap_manager_.batch_delete(segment_ids);
    // update filter cache map
    filter_cache_.release_for_segments(segment_ids, level_0_segment_ids);
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
    bool can_adjust = heap_manager_.try_modify(result);
    if (can_adjust) {
        std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
        std::set<uint32_t> empty_level_0_segment_ids; // no level 0 segment in heaps, dont worry
        std::set<uint32_t> empty_failed_segment_ids; // force to update segments' filter units group, so dont worry for cache space
        segment_units_num_recorder.insert(std::make_pair(result.enable_segment_id, result.enable_segment_next_units_num));
        segment_units_num_recorder.insert(std::make_pair(result.disable_segment_id, result.disable_segment_next_units_num));
        filter_cache_.update_for_segments(segment_units_num_recorder, true, empty_level_0_segment_ids, empty_failed_segment_ids);
    }
    return can_adjust;
}

void FilterCacheManager::insert_segments(std::vector<uint32_t>& merged_segment_ids, std::vector<uint32_t>& new_segment_ids,
                                         std::map<uint32_t, std::unordered_map<uint32_t, double>>& inherit_infos_recorder,
                                         std::map<uint32_t, uint16_t>& level_recorder, const uint32_t& level_0_base_count,
                                         std::map<uint32_t, std::vector<RangeRatePair>>& segment_ranges_recorder) {
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    std::map<uint32_t, uint32_t> approximate_counts_recorder;
    std::set<uint32_t> failed_segment_ids;
    std::vector<FilterCacheHeapItem> new_segment_items;
    std::set<uint32_t> old_level_0_segment_ids, new_level_0_segment_ids;
    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::sort(merged_segment_ids.begin(), merged_segment_ids.end());
    std::sort(new_segment_ids.begin(), new_segment_ids.end());

    // pick up merged or new level 0 segments
    // assume level_recorder keys set equals to merged_segment_ids + new_segment_ids
    assert(new_segment_ids.size() == 0 || merged_segment_ids.size() + new_segment_ids.size() == level_recorder.size());
    auto level_it = level_recorder.begin();
    size_t merged_idx = 0, new_idx = 0;
    while (level_it != level_recorder.end()) {
        if (merged_idx < merged_segment_ids.size() && level_it->first == merged_segment_ids[merged_idx]) {
            if (level_it->second == 0) {
                old_level_0_segment_ids.insert(level_it->first);
            }
            merged_idx ++;
        } else if (new_idx < new_segment_ids.size() && level_it->first == new_segment_ids[new_idx]) {
            if (level_it->second == 0) {
                new_level_0_segment_ids.insert(level_it->first);
                segment_units_num_recorder.insert(std::make_pair(level_it->first, MAX_UNITS_NUM));
            } else {
                // not a level 0 segment, set default units num
                segment_units_num_recorder.insert(std::make_pair(level_it->first, DEFAULT_UNITS_NUM));
            }
            new_idx ++;
        } 
        level_it ++;
    }

    if (!is_ready_) {
        // if is_ready_ is false, no need to enable two-heaps adjustment, remember to update is_ready_ in the end
        // remove merged segments' units in filter cache and nodes in filter heaps
        heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // inherit merged segments' counts to new segments' counts
        // ensure that new segments that are not in inherit_infos_recorder keys set are only level 0 segments
        inherit_count_recorder(merged_segment_ids, new_segment_ids, level_0_base_count, inherit_infos_recorder);
        estimate_counts_for_all(approximate_counts_recorder);

        // insert units into filter cache
        filter_cache_.enable_for_segments(segment_units_num_recorder, false, new_level_0_segment_ids, failed_segment_ids);
        
        // insert nodes into filter heaps
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id)) {
                // no need to insert level 0 segment nodes into heap
                continue;
            } else if (failed_segment_ids.count(new_segment_id)) {
                // failed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                                                                   0, 0, units_num));
            } else {
                // succeed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                                                                   units_num, 0, units_num));
            }
        }
        heap_manager_.batch_upsert(new_segment_items);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // is_ready_ is true, then we will not update is_ready_, that means is_ready_ will be always true
        // remove merged segments' units in filter cache and nodes in filter heaps
        heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // inherit merged segments' counts to new segments' counts
        // ensure that new segments that are not in inherit_infos_recorder keys set are only level 0 segments
        inherit_count_recorder(merged_segment_ids, new_segment_ids, level_0_base_count, inherit_infos_recorder);
        estimate_counts_for_all(approximate_counts_recorder);

        // predict units num for new non level 0 segments and update segment_units_num_recorder
        std::vector<std::vector<uint32_t>> pred_datas;
        std::vector<uint32_t> pred_segment_ids;
        std::vector<uint16_t> pred_results;
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id)) {
                // no need to predict for level 0 segments
                continue;
            } else {
                pred_segment_ids.emplace_back(new_segment_id);

                std::vector<uint32_t> pred_data;
                pred_data.emplace_back(level_recorder[new_segment_id]);
                for (RangeRatePair& pair : segment_ranges_recorder[new_segment_id]) {
                    assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                    pred_data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * pair.rate_in_segment));
                    pred_data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * buckets[pair.range_id].hotness_));
                }
                pred_datas.emplace_back(pred_data);
            }
        }
        assert(pred_datas.size() == pred_segment_ids.size());
        clf_model_.make_predict(pred_datas, pred_results);
        assert(pred_datas.size() == pred_results.size());
        size_t pred_idx = 0;
        while (pred_idx < pred_segment_ids.size() && pred_idx < pred_results.size()) {
            segment_units_num_recorder[pred_segment_ids[pred_idx]] = pred_results[pred_idx];
            pred_idx = pred_idx + 1;
        }

        // insert units into filter cache
        filter_cache_.enable_for_segments(segment_units_num_recorder, false, new_level_0_segment_ids, failed_segment_ids);

        // insert nodes into filter heaps
        for (uint32_t& new_segment_id : new_segment_ids) {
            if (new_level_0_segment_ids.count(new_segment_id)) {
                // no need to insert level 0 segment nodes into heap
                continue;
            } else if (failed_segment_ids.count(new_segment_id)) {
                // failed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                                                                   0, 0, units_num));
            } else {
                // succeed to insert filter units
                uint16_t units_num = segment_units_num_recorder[new_segment_id];
                new_segment_items.emplace_back(FilterCacheHeapItem(new_segment_id, approximate_counts_recorder[new_segment_id],
                                                                   units_num, 0, units_num));
            }
        }
        heap_manager_.batch_upsert(new_segment_items);
    }
}

void FilterCacheManager::delete_segments(std::vector<uint32_t>& merged_segment_ids, std::map<uint32_t, uint16_t>& level_recorder) {
    std::set<uint32_t> old_level_0_segment_ids;
    std::sort(merged_segment_ids.begin(), merged_segment_ids.end());

    // level_recorder is a copy of global level_recorder
    assert(merged_segment_ids.size() == level_recorder.size());
    auto level_it = level_recorder.begin();
    size_t merged_idx = 0;
    while (level_it != level_recorder.end()) {
        assert(merged_idx < merged_segment_ids.size() && level_it->first == merged_segment_ids[merged_idx]);
        if (merged_idx < merged_segment_ids.size() && level_it->first == merged_segment_ids[merged_idx]) {
            if (level_it->second == 0) {
                old_level_0_segment_ids.insert(level_it->first);
            }
            merged_idx ++;
        }
        level_it ++;
    }

    if (!is_ready_) {
        // if is_ready_ is false, no need to enable two-heaps adjustment, remember to update is_ready_ in the end
        // remove merged segments' units in filter cache and nodes in filter heaps
        heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // is_ready_ is true, then we will not update is_ready_, that means is_ready_ will be always true
        // remove merged segments' units in filter cache and nodes in filter heaps
        heap_manager_.batch_delete(merged_segment_ids);
        filter_cache_.release_for_segments(merged_segment_ids, old_level_0_segment_ids);
    }
}

void FilterCacheManager::move_segments(std::vector<uint32_t>& moved_segment_ids,
                                       std::map<uint32_t, uint16_t>& old_level_recorder,
                                       std::map<uint32_t, uint16_t>& move_level_recorder,
                                       std::map<uint32_t, std::vector<RangeRatePair>>& move_segment_ranges_recorder) {
    std::unordered_map<uint32_t, uint16_t> segment_units_num_recorder;
    std::map<uint32_t, uint32_t> approximate_counts_recorder;
    std::vector<FilterCacheHeapItem> new_segment_items;
    std::set<uint32_t> old_level_0_segment_ids;
    std::vector<Bucket> buckets = heat_buckets_.buckets();
    std::sort(moved_segment_ids.begin(), moved_segment_ids.end());

    // pick up merged or new level 0 segments, but this type of compaction must not move to level 0,
    // so we may only move level 0 to level below
    assert(moved_segment_ids.size() == old_level_recorder.size());
    assert(moved_segment_ids.size() == move_level_recorder.size());
    assert(moved_segment_ids.size() == move_segment_ranges_recorder.size());
    auto level_it = old_level_recorder.begin();
    size_t moved_idx = 0, new_idx = 0;
    while (level_it != old_level_recorder.end()) {
        assert(moved_idx < moved_segment_ids.size() && level_it->first == moved_segment_ids[moved_idx]);
        if (moved_idx < moved_segment_ids.size() && level_it->first == moved_segment_ids[moved_idx]) {
            if (level_it->second == 0) {
                old_level_0_segment_ids.insert(level_it->first);
            }
            segment_units_num_recorder.insert(std::make_pair(level_it->first, DEFAULT_UNITS_NUM));
            // actually, we cannot move segments to level 0 in trivial move compaction (only flushing do this).
            moved_idx ++;
        }
        level_it ++;
    }

    if (!is_ready_) {
        // firstly, delete moved segments
        heap_manager_.batch_delete(moved_segment_ids);
        filter_cache_.release_for_segments(moved_segment_ids, old_level_0_segment_ids);

        // inherit these segments' count
        for (uint32_t& segment_id : moved_segment_ids) {
            auto last_it = last_count_recorder_.find(segment_id);
            auto current_it = current_count_recorder_.find(segment_id);
            if (last_it != last_count_recorder_.end()) {
                last_it->second = INHERIT_REMAIN_FACTOR * (last_it->second);
            }
            if (current_it != current_count_recorder_.end()) {
                current_it->second = INHERIT_REMAIN_FACTOR * (current_it->second);
            }
        }
        estimate_counts_for_all(approximate_counts_recorder);

        // insert units into filter cache
        std::set<uint32_t> empty_new_level_0_segment_ids, empty_failed_segment_ids;
        filter_cache_.enable_for_segments(segment_units_num_recorder, true, empty_new_level_0_segment_ids, empty_failed_segment_ids);
        
        // insert nodes into filter heaps
        for (uint32_t& segment_id : moved_segment_ids) {
            assert(move_level_recorder[segment_id] > 0);
            uint16_t units_num = segment_units_num_recorder[segment_id];
            new_segment_items.emplace_back(FilterCacheHeapItem(segment_id, approximate_counts_recorder[segment_id],
                                                               units_num, 0, units_num));
        }
        heap_manager_.batch_upsert(new_segment_items);

        // remember to update is_ready_
        if (filter_cache_.is_ready()) {
            is_ready_ = true;
        }
    } else {
        // firstly, delete moved segments
        heap_manager_.batch_delete(moved_segment_ids);
        filter_cache_.release_for_segments(moved_segment_ids, old_level_0_segment_ids);

        // inherit these segments' count
        for (uint32_t& segment_id : moved_segment_ids) {
            auto last_it = last_count_recorder_.find(segment_id);
            auto current_it = current_count_recorder_.find(segment_id);
            if (last_it != last_count_recorder_.end()) {
                last_it->second = INHERIT_REMAIN_FACTOR * (last_it->second);
            }
            if (current_it != current_count_recorder_.end()) {
                current_it->second = INHERIT_REMAIN_FACTOR * (current_it->second);
            }
        }
        estimate_counts_for_all(approximate_counts_recorder);

        // predict units num for new non level 0 segments and update segment_units_num_recorder
        std::vector<std::vector<uint32_t>> pred_datas;
        std::vector<uint32_t> pred_segment_ids;
        std::vector<uint16_t> pred_results;
        for (uint32_t& segment_id : moved_segment_ids) {
            assert(move_level_recorder[segment_id] > 0);
            pred_segment_ids.emplace_back(segment_id);

            std::vector<uint32_t> pred_data;
            pred_data.emplace_back(move_level_recorder[segment_id]);
            for (RangeRatePair& pair : move_segment_ranges_recorder[segment_id]) {
                assert(pair.range_id >= 0 && pair.range_id < buckets.size());
                pred_data.emplace_back(uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * pair.rate_in_segment));
                pred_data.emplace_back(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * buckets[pair.range_id].hotness_));
            }
        }
        assert(pred_datas.size() == pred_segment_ids.size());
        clf_model_.make_predict(pred_datas, pred_results);
        assert(pred_datas.size() == pred_results.size());
        size_t pred_idx = 0;
        while (pred_idx < pred_segment_ids.size() && pred_idx < pred_results.size()) {
            segment_units_num_recorder[pred_segment_ids[pred_idx]] = pred_results[pred_idx];
            pred_idx = pred_idx + 1;
        }

        // insert units into filter cache
        std::set<uint32_t> empty_new_level_0_segment_ids, empty_failed_segment_ids;
        filter_cache_.enable_for_segments(segment_units_num_recorder, true, empty_new_level_0_segment_ids, empty_failed_segment_ids);

        // insert nodes into filter heaps
        for (uint32_t& segment_id : moved_segment_ids) {
            assert(move_level_recorder[segment_id] > 0);
            uint16_t units_num = segment_units_num_recorder[segment_id];
            new_segment_items.emplace_back(FilterCacheHeapItem(segment_id, approximate_counts_recorder[segment_id],
                                                               units_num, 0, units_num));
        }
        heap_manager_.batch_upsert(new_segment_items);
    }
}

}