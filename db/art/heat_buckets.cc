#include "heat_buckets.h"
#include <fstream>
#include <iostream>

namespace ROCKSDB_NAMESPACE {
std::vector<std::string> HeatBuckets::seperators_;
std::vector<Bucket> HeatBuckets::buckets_;
uint32_t HeatBuckets::current_cnt_; // current get count in this period
std::vector<std::unique_ptr<std::mutex>> HeatBuckets::mutex_ptrs_;
std::mutex HeatBuckets::cnt_mutex_;
std::mutex HeatBuckets::sample_mutex_;
bool HeatBuckets::is_ready_; // identify whether HeatBuckets ready for hit
SamplesPool HeatBuckets::samples_; 
bool HeatBuckets::updated_; // prevent from updating hotness more than once in a short time


Bucket::Bucket() {
    hit_cnt_ = 0;
    hotness_ = 0;
    // keys_.clear();
}

Bucket::~Bucket() {
    return; // destroy nothing
}

void Bucket::update(const double& alpha, const uint32_t& period_cnt) {
    // mutex_.lock();
    hotness_ = alpha * hotness_ + 
                (1 - alpha) * double(hit_cnt_) / double(period_cnt);
    hit_cnt_ = 0; // remember to reset counter
    // keys_.clear();
    // mutex_.unlock(); // remember to unlock!!!
}

void Bucket::hit() {
    // mutex_.lock();
    hit_cnt_ += 1;
    // keys_.insert(key);
    // mutex_.unlock(); // remember to unlock!!!
}

HeatBuckets::HeatBuckets() {
    seperators_.resize(0);
    buckets_.resize(0);
    current_cnt_ = 0;
    mutex_ptrs_.resize(0);
    is_ready_ = false;
    samples_.clear(); 
    updated_ = false;
}

HeatBuckets::~HeatBuckets() {
    return; // destroy nothing
}

void HeatBuckets::debug() {
    std::cout << "[DEBUG] total cnt in this period: " << current_cnt_ << std::endl;
    for (auto& bucket : buckets_) {
        std::cout << "[DEBUG] ";
        std::cout << "bucket hotness : " << bucket.hotness_;
        std::cout << ", bucket hit cnt : " << bucket.hit_cnt_;
        // std::cout << ", bucket keys cnt : " << bucket.keys_cnt();
        std::cout << std::endl;
    }
}

void HeatBuckets::update() {
    // mark already updated, after current_cnt_ more than PERIOD_COUNT / MAGIC_FACTOR, updated_ will be reset to false;
    // we need guarantee that in one period (one constant time span), db gets are much larger than PERIOD_COUNT / MAGIC_FACTOR;
    // usually in server, exec get requests PERIOD_COUNT / MAGIC_FACTOR times only account for a very very short time.
    if (updated_) {
        return;
    }

    updated_ = true; 
    
    assert(mutex_ptrs_.size() == buckets_.size());
    for (size_t i=0; i<mutex_ptrs_.size(); i++) {
        mutex_ptrs_[i]->lock();
    }

    // TODO: use multiple threads to update hotness of all buckets
    for (size_t i=0; i<buckets_.size(); i++) {
        buckets_[i].update(BUCKETS_ALPHA, current_cnt_);
        mutex_ptrs_[i]->unlock();
    }
    // remember to reset current_cnt_ counter
    current_cnt_ = 0;
}

uint32_t HeatBuckets::locate(const std::string& key) {
    // we use locate method to locate the key range for one key
    // reminded one key range -> [lower seperator, upper seperator)
    // if we locate key k to idx i, then seperator i <= k < seperator i+1
    // equal to k in key range i
    uint32_t left = 0, right = seperators_.size()-1;
    while (left < right - 1){				  
        uint32_t mid = left + ((right-left) / 2);
        if (seperators_[mid] > key) {
            right = mid;	  		 
        } else if (seperators_[mid] <= key) {
            left = mid;		        
        }
    }
    return left;
}

void HeatBuckets::hit(const std::string& key, const bool& signal) {
    assert(is_ready_);
    // use binary search to find index i, making seperators_[i] <= key and seperators_[i+1] > i
    // reminding we have set border guard, so dont worry about out of bounds error
    // after we find the index i, we call buckets_[i].hit(), then add 1 to current_cnt_
    // if current_cnt_ >= period_cnt_, call update() to update hotness of all buckets and reset current cnt counter

    uint32_t idx = 0;
    // last element is border guard
    // means last element always bigger than key
    // first element is border guard
    // means first element always smaller than key

    // linear search version
    /* 
    while (seperators_[index+1] <= key) {
        index += 1;
    }
    */
    idx = locate(key);

    // std::cout << "debug seperators_ size : " << seperators_.size() << std::endl;
    // std::cout << "debug buckets_ size : " << buckets_.size() << std::endl;
    // std::cout << "debug mutex_ptrs_ size : " << mutex_ptrs_.size() << std::endl;
    // std::cout << "debug period_cnt_ : " << period_cnt_ << std::endl;
    // std::cout << "debug alpha_ : " << alpha_ << std::endl;
    assert(buckets_.size() == mutex_ptrs_.size());
    assert(idx >= 0 && idx < buckets_.size());
    assert(seperators_[idx] <= key && key < seperators_[idx+1]);
    
    mutex_ptrs_[idx]->lock();
    buckets_[idx].hit(); // mutex only permits one write opr to one bucket
    mutex_ptrs_[idx]->unlock();

    cnt_mutex_.lock();
    current_cnt_ += 1;
   
    // use updated_ to prevent from updating hotness in a very short time span (due to multi-threads operation)
    if (signal && !updated_) {
        // debug();
        update();
    }
    cnt_mutex_.unlock();

    // remember to reset updated_ to false
    if (updated_ && current_cnt_ >= PERIOD_COUNT / MAGIC_FACTOR) {
        updated_ = false; 
    }
}

SamplesPool::SamplesPool() {
    samples_cnt_ = 0;
    pool_.resize(0);
    filter_.clear();

    // because put opt will input duplicated keys, we need to guarantee SAMPLES_MAXCNT much larger than SAMPLES_LIMIT
    // however std::set only remain deduplicated keys
    // to collect good samples for previous put keys, we need a larger SAMPLES_MAXCNT
    assert(SAMPLES_MAXCNT >= MAGIC_FACTOR * SAMPLES_LIMIT); 
}

void SamplesPool::clear() {
    samples_cnt_ = 0;
    pool_.resize(0);
    filter_.clear();
}

void SamplesPool::sample(const std::string& key) {
    assert(pool_.size() == filter_.size());
    // if already in pool, return
    if (is_sampled(key)) {
        return;
    }
    // pool not full
    if (!is_full()) {
        pool_.push_back(key);
        filter_.insert(key);
    } 
    // pool is full
    else {
        // need to generate random integer in [0, old samples_cnt_] (equal to [0, old samples_cnt_ + 1))
        // new samples_cnt_ = old samples_cnt_ + 1
        // if you want random integer in [a, b], use (rand() % (b-a+1))+a;
        srand((unsigned)time(NULL));
        uint32_t idx = (rand() % (samples_cnt_ + 1)) + 0;
        assert(idx <= samples_cnt_ && idx >= 0);
        // idx in [0, samples_limit_)
        // pool_ size may lightly more than samples_limit_;
        if (idx < pool_.size()) {
            // remove old key
            std::string old_key = pool_[idx];
            filter_.erase(old_key);

            // update new key
            pool_[idx] = key;
            filter_.insert(key);
        }
    }
    assert(pool_.size() == filter_.size());

    // remember to update samples_cnt_
    samples_cnt_ += 1;
}

void SamplesPool::prepare() {
    std::string key_min = "user"; // defined min key for YCSB
    std::string key_max = pool_[pool_.size()-1] + pool_[pool_.size()-1];
    if (!is_ready()) {
        return;
    }
    sort(pool_.begin(), pool_.end());
    // add border guard
    pool_.emplace(pool_.begin(), key_min);
    pool_.emplace_back(key_max);
}

void SamplesPool::divide(const uint32_t& k, std::vector<std::string>& dst) { 
    // reminded we already add border guard to pool vector
    std::string key_min = pool_[0]; // defined min key for YCSB
    std::string key_max = pool_[pool_.size()-1];

    dst.resize(0);
    dst.emplace_back(key_min);

    // reminded we already add border guard to pool vector
    // border guard in idx 0 and idx pool_.size()-1
    uint32_t idx = 1;
    while (idx < pool_.size() - 1) {
        dst.emplace_back(pool_[idx]);
        idx += k;
    }

    dst.emplace_back(key_max);
}


uint32_t SamplesPool::locate(const std::string& key) {
    // pool must be sorted 
    // and we need to add border guard to pool
    // after that, we can use locate(key)
    uint32_t left = 0, right = pool_.size()-1;
    while (left < right - 1){				  
        uint32_t mid = left + ((right-left) / 2);
        if (pool_[mid] > key) {
            right = mid;	  		 
        } else if (pool_[mid] <= key) {
            left = mid;		        
        }
    }
    return left;
}

uint32_t SamplesPool::determine_k(std::vector<std::vector<std::string>>& segments) {
    // already add border guard to pool
    uint32_t k = pool_.size() - 2;
    // if segments is empty, use default k to debug
    if (segments.empty()) {
        k = (pool_.size() - 2) / DEFAULT_BUCKETS_NUM;  
    }
    assert(k > 1);
    for (auto& segment : segments) {
        assert(segment.size() == 2);
        assert(segment[0] < segment[1]);
        uint32_t span = locate(segment[1]) - locate(segment[0]);
        
        assert(span > 1);
        if (k > span) k = span;
    }
    // std::cout << "[DEBUG] samples divided with span k : " << k << std::endl;
    return k;
} 

void HeatBuckets::sample(const std::string& key, std::vector<std::vector<std::string>>& segments) {
    if (samples_.is_ready()) {
        return;
    }
    sample_mutex_.lock();
    if (!samples_.is_ready()) {
        samples_.sample(key);
        if (samples_.is_ready()) {
            init(segments);
        }
    }
    sample_mutex_.unlock();
}

void HeatBuckets::init(std::vector<std::vector<std::string>>& segments) {
    // compute proper k and determine key ranges
    samples_.prepare();
    uint32_t k = samples_.determine_k(segments);
    samples_.divide(k, seperators_);

    // std::cout << "[DEBUG] show key ranges below: " << std::endl;
    for (size_t i=0; i<seperators_.size()-1; i++) {
        assert(seperators_[i] < seperators_[i+1]);
        // std::cout << "[DEBUG] key range " << i+1;
        // std::cout << ": " << seperators_[i];
        // std::cout << "  --  " << seperators_[i+1];
        // std::cout << std::endl;
    }

    // init other vars in HeatBuckets
    current_cnt_ = 0;
    buckets_.resize(seperators_.size()-1);
    mutex_ptrs_.resize(0);
    for (uint32_t i=0; i<buckets_.size(); i++) {
        mutex_ptrs_.emplace_back(std::unique_ptr<std::mutex>(new std::mutex()));
    }
    assert(mutex_ptrs_.size() == buckets_.size());
    assert(seperators_.size() == buckets_.size()+1);

    is_ready_ = true;

    // debug
    // std::cout << "[DEBUG] heat buckets size: " << buckets_.size() << std::endl;
    // std::cout << "[DEBUG] key ranges init" << std::endl;
}
}