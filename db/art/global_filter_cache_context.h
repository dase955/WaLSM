
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "db/art/filter_cache_client.h"

namespace ROCKSDB_NAMESPACE {
// TODO: add necessary filter cache info structures
extern rocksdb::FilterCacheClient
    global_filter_cache;  // already contain FilterCacheManager

// TODO: mutex for updating these recorders below
//       will be locked when updating these recorders below, and unlock after
//       updating ends
extern std::mutex global_filter_cache_recorders_mutex;

// these global recorders need to be latest after every flush or compaction:
// std::map<uint32_t, uint16_t>* level_recorder_
// std::map<uint32_t, std::vector<RangeRatePair>>* segment_ranges_recorder_
// std::map<uint32_t, uint32_t>* unit_size_recorder_
// you may need filter_cache_.range_seperators() to receive key range seperators
// exactly, if key k < seperators[i+1] and key k >= seperators[i], then key k
// hit key range i HeatBuckets::locate(const std::string& key) will tell you how
// to binary search corresponding key range for one key

// segment_info_recorder save every segments' min key and max key
// but we only need to pass empty segment_info_recorder now
// TODO: it should contain all levels segments' min key and max key, then pass
// to filter cache client, but not used now this recorder will help decide the
// key ranges' num, but it dont work in current work you can try to modify macro
// DEFAULT_BUCKETS_NUM to decide the key ranges' num
extern std::unordered_map<uint32_t, std::vector<std::string>>
    global_segment_info_recorder;

// record every alive segments' level
// TODO: need to be latest all the time
extern std::map<uint32_t, uint16_t> global_level_recorder;

// record features num of every segments
// we choose max features num to define model feature num
// if you want to use a default features num, set MAX_FEATURES_NUM to non-zero
// value then do not insert any entry into this vector later
// TODO: we dont use this vector, so we set MAX_FEATURES_NUM to non-zero value
extern std::vector<uint16_t> global_features_nums_except_level_0;

// should be based level 0 visit cnt in a total long period
// simply we set level_0_base_count to 0, and use macro INIT_LEVEL_0_COUNT
// we can set this macro to ( PERIOD_COUNT * TRAIN_PERIODS ) * ( level 0 sorted
// runs num ) / ( max level 0 segments num )
// TODO: modify INIT_LEVEL_0_COUNT to proper value
extern uint32_t global_level_0_base_count;

// record interacting ranges and their rates of alive segments
// TODO: should be latest all the time
extern std::map<uint32_t, std::vector<RangeRatePair>>
    global_segment_ranges_recorder;

// every segment's filter unit size is the same
// this recorder should hold all alive segment
// simply, you can also use default macro DEFAULT_UNIT_SIZE for all segments,
// just leave this recorder empty
// TODO: modify DEFAULT_UNIT_SIZE
extern std::map<uint32_t, uint32_t> global_unit_size_recorder;

}  // namespace ROCKSDB_NAMESPACE