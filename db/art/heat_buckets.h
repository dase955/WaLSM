#pragma once
#include <mutex>
#include <vector>
#include <string>
#include <set>
#include "macros.h"
#include <memory>
#include <cassert>
#include <cstdlib>
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

class Bucket;
class HeatBuckets;
class SamplesPool;

class Bucket {
public:
    double hotness_;
    uint32_t hit_cnt_;

    Bucket();
    ~Bucket();

    // when one time period end, update hotness_, h_(i+1) = alpha * h_i + hit_cnt_ / period_cnt
    void update(const double& alpha, const uint32_t& period_cnt); 
    void hit(); 
};


/*
 first sample put keys using reservoir sampling. 
 If we collect enough keys, determine the common key num (k) for every key group
 start with idx 0, add k continuously, get 0, k, 2k, ...
 set KEY_MIN, samples[0], samples[k], samples[2k], ..., KEY_MAX as seperators
 guarenteed that KEY_MIN < all keys and KEY_MAX > all keys
 then we define key ranges (KEY_MIN, samples[0]), [samples[1], samples[2]), [samples[2], samples[3]), ..., [..., KEY_MAX)
 one heat bucket corresponding to one key range
 compute and update hotness of all heat buckets
*/
class HeatBuckets {
private:
    // TODO: mutex can be optimized
    static std::vector<std::string> seperators_;
    static std::vector<Bucket> buckets_;
    static uint32_t current_cnt_; // current get count in this period
    static std::vector<std::unique_ptr<std::mutex>> mutex_ptrs_;
    static std::mutex cnt_mutex_;
    static std::mutex sample_mutex_;
    static bool is_ready_; // identify whether HeatBuckets ready for hit
    static SamplesPool samples_; 
    static bool updated_;
    
public:
    HeatBuckets();
    ~HeatBuckets();

    uint32_t locate(const std::string& key); // helper func: locate which bucket hitted by this key

    const bool& is_ready() { return is_ready_; }
    std::vector<std::string>& seperators() { return seperators_; }
    std::vector<Bucket>& buckets() { return buckets_; }
    void sample(const std::string& key, std::vector<std::vector<std::string>>& segments); // before init buckets, we need to sample keys;
    // input segment-related key range (segments), will use them when SamplesPool ready. 

    void init(std::vector<std::vector<std::string>>& segments); // if sample enough keys, ready to init heatbuckets

    void update(); // update hotness value of all buckets
    void hit(const std::string& key, const bool& signal); // one key only hit one bucket (also mean only hit one key range)
    // if signal is true, update hotness
    void debug(); // output debug message in standard output
};

class SamplesPool {
private:
    std::vector<std::string> pool_; // using set to guarantee only store deduplicated samples
    std::set<std::string> filter_; // used to check whether new key already exist in pool
    uint32_t samples_cnt_; // current sample tries num, need to update after every try
public:
    SamplesPool();

    ~SamplesPool() { return; }

    void clear();

    // we can modify SAMPLES_MAXCNT to control the moment that starts init heat buckets
    bool is_ready() { return samples_cnt_ >= SAMPLES_MAXCNT; }
    bool is_full() { return pool_.size() >= SAMPLES_LIMIT; }
    bool is_sampled(const std::string& key) { return filter_.count(key) > 0; }

    void sample(const std::string& key);

    void prepare();

    // need call prepare() before
    // generate seperators
    void divide(const uint32_t& k, std::vector<std::string>& dst);

    // determine k based on low-level segments' key range
    uint32_t determine_k(std::vector<std::vector<std::string>>& segments);
    uint32_t locate(const std::string& key); // helper func when determine k
};

}