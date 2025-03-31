#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <string>
#include <cassert>
#include <iostream>
#include "macros.h"

// dataset data point format: 
// every data point accounts for one segment
// supposed that considering r key range
// every key range have id and hotness ( see heat_buckets )
// so data point features format : 
// LSM-Tree level, Key Range 1 rate, Key Range 1 hotness, Key Range 2 rate, Key Range 2 hotness, ..., Key Range r rate, Key Range r hotness
// we also need to append best units num (from solving programming problem) and visit count to every row
// so in data csv, one row would be like:
// LSM-Tree level, key range 1 rate, key range 1 hotness, ..., best units num (for the segment), visit count to this segment (in last long peroid)
// for example, assume that segment 1 can be devide into key range 3 (50% keys), key range 4 (30% keys), key range 6 (20% keys)
// sort these key ranges by their keys rate (e.g. key range 3 rate = 50%) and set feature num to 7
// data format can be like:
// 5 (segment 1 level in LSM-Tree), 50% (key range 3 rate), 5234 (key range 3 hotness), 30% (key range 4 rate), 2222 (key range 4 hotness), 20% (key range 6 rate), 11111 (key range 6 hotness)
// remind that heat_buckets recorded hotness is double type, 
// because data feature only accept uint32_t type,
// we use uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * hotness) to closely estimate its hotness value
// also use uint32_t(RATE_SIGNIFICANT_DIGITS_FACTOR * rate) to closely estimate its rate in this segment

namespace ROCKSDB_NAMESPACE {

struct RangeRatePair;
class ClfModel;

bool RangeRatePairLessorComparor(const RangeRatePair& pair_1, const RangeRatePair& pair_2);
bool RangeRatePairGreatorComparor(const RangeRatePair& pair_1, const RangeRatePair& pair_2);

struct RangeRatePair {
    uint32_t range_id;
    double rate_in_segment;
};

inline bool RangeRatePairLessorComparor(const RangeRatePair& pair_1, const RangeRatePair& pair_2) {
    return pair_1.rate_in_segment < pair_2.rate_in_segment;
}

inline bool RangeRatePairGreatorComparor(const RangeRatePair& pair_1, const RangeRatePair& pair_2) {
    return pair_1.rate_in_segment > pair_2.rate_in_segment;
}

class ClfModel {
private:
    static uint16_t feature_num_; // model input features num
    static std::string dataset_name_; // dataset csv file name
    static std::string dataset_path_; // path to save dataset csv file
    static std::string host_, port_; // lightgbm server connection
    static size_t buffer_size_; // socket receive buffer max size
public:
    // init member vars
    ClfModel() {
        feature_num_ = 0;
        dataset_name_ = DATASET_NAME;
        // MODEL_PATH must end with '/'
        dataset_path_ = MODEL_PATH;
        dataset_path_ = dataset_path_ + DATASET_NAME;
        host_ = HOST; port_ = PORT;
        buffer_size_ = BUFFER_SIZE;
    }

    // check whether ready, only need check feature_nums_ now
    bool is_ready() { return feature_num_ > 0; }

    // make ready for training, only need init feature_nums_ now
    // when first call ClfModel, we need to use current segments information to init features_num_
    // we can calcuate feature nums for every segment, 
    // feature num = level feature num (1) + 2 * num of key ranges 
    // we set features_num_ to largest feature num
    void make_ready(std::vector<uint16_t>& features_nums) { 
        if (features_nums.empty()) {
            feature_num_ = 41; // debug feature num, see ../lgb_server files
        } else {
            // we may limit feature_num_ because of the socket transmit size limit is 1024 bytes
            // so feature_num_ may be limit to at most about 3 * 30 + 1 = 91
            feature_num_ = *max_element(features_nums.begin(), features_nums.end()); 
            if (feature_num_ > MAX_FEATURES_NUM) {
                feature_num_ = MAX_FEATURES_NUM;
            }
        }

        // std::cout << "[DEBUG] ClfModel ready, feature_num_: " << feature_num_ << std::endl;
    }

    ~ClfModel() {
        return; // do nothing
    }

    // resize data point features
    void prepare_data(std::vector<uint32_t>& data) { 
        // at least level feature and one key range, so data size always >= 3
        assert(data.size() >= 3);
        data.resize(feature_num_, 0); 
    }

    // resize every data point and write to csv file for training
    void write_debug_dataset();
    void write_real_dataset(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts);
    void write_dataset(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts);

    // write dataset then send msg to train new model in LightGBM server side
    void make_train(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts);

    // predict
    void make_predict_samples(std::vector<std::vector<uint32_t>>& datas);
    void make_real_predict(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& preds);
    void make_predict(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& preds);
};

}