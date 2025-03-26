#include "clf_model.h"
#include <csv2/writer.hpp>
#include <csv2/reader.hpp>
#include <libsocket/exception.hpp>
#include <libsocket/inetclientstream.hpp>
#include <vector>
#include <map>
#include <random>
#include <chrono>

namespace ROCKSDB_NAMESPACE {

uint16_t ClfModel::feature_num_; 
std::string ClfModel::dataset_name_;
std::string ClfModel::dataset_path_;
std::string ClfModel::host_, ClfModel::port_;
size_t ClfModel::buffer_size_;

void ClfModel::write_debug_dataset() {
    assert(feature_num_ > 0);
    // ready for writer
    std::ofstream stream(dataset_path_);
    csv2::Writer<csv2::delimiter<','>> writer(stream);

    // init hotness values
    std::map<uint32_t, double> hotness_map;
    double base_hotness = 0.01;
    for (int i = 0; i < 200; i ++) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + base_hotness;
        hotness_map[i] = r;
    }

    // init header vector
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header;
    header.emplace_back("Level");
    for (int i = 0; i < 20; i ++) {
        header.emplace_back("Rate_" + std::to_string(i));
        header.emplace_back("Hotness_" + std::to_string(i));
    }
    header.emplace_back("Target");
    header.emplace_back("Count");
    rows.emplace_back(header);

    // ready for shuffling
    std::vector<uint32_t> ids;
    for(int i = 0; i < 200; i ++) {
        ids.emplace_back(i);
    }

    // generate values
    for (int i = 0; i < 1000; i ++) {
        // std::vector<double> value;
        std::vector<std::string> values;
        uint32_t level = i / 200;
        uint16_t target = 5 - level;
        uint32_t count = (1000 - i) * 100;
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if (r > 0.10 * level) {
            target -= 1;
            count -= 100;
        }

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
        values.emplace_back(std::to_string(level));
        for (int j = 0; j < 20; j ++) {
            values.emplace_back(std::to_string(uint32_t(ids[j] * 0.005 * RATE_SIGNIFICANT_DIGITS_FACTOR)));
            values.emplace_back(std::to_string(uint32_t(HOTNESS_SIGNIFICANT_DIGITS_FACTOR * hotness_map[ids[j]])));
        }
        values.emplace_back(std::to_string(target)); // Target column
        values.emplace_back(std::to_string(count)); // Count column
        assert(values.size() == feature_num_ + 2);
        rows.emplace_back(values);
    }

    writer.write_rows(rows);
    stream.close();
}

void ClfModel::write_real_dataset(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts) {
    assert(feature_num_ > 0);
    // tags is real class of all segments, 
    // we also need to write these tags to dataset besides features
    assert(datas.size()==tags.size());
    assert(datas.size()==get_cnts.size());
    // ready for writer
    std::ofstream stream(dataset_path_);
    csv2::Writer<csv2::delimiter<','>> writer(stream);

    // init csv header vector
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header;
    uint16_t ranges_num = (feature_num_ - 1) / 2;
    header.emplace_back("Level");
    for (int i = 0; i < ranges_num; i ++) {
        header.emplace_back("Rate_" + std::to_string(i));
        header.emplace_back("Hotness_" + std::to_string(i));
    }
    // remind that targeted class is in csv Target column
    // corresponding to code of lgb.py in ../models dir
    header.emplace_back("Target");
    header.emplace_back("Count");
    rows.emplace_back(header);

    std::vector<std::string> values;
    size_t idx = 0;
    for (std::vector<uint32_t>& data : datas) {
        // resize features vector to size feature_num_
        prepare_data(data);
        values.clear();
        for (uint32_t& value : data) {
            values.emplace_back(std::to_string(value));
        }
        // remember to write real tag and get cnt to dataset
        values.emplace_back(std::to_string(tags[idx]));
        values.emplace_back(std::to_string(get_cnts[idx++]));
        assert(values.size() == feature_num_ + 2);
        rows.emplace_back(values);
    }

    writer.write_rows(rows);
    stream.close();
}

void ClfModel::write_dataset(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts) {
    assert(feature_num_ > 0);
    if (datas.empty()) {
        write_debug_dataset();
        // dataset_cnt_ += 1;
        return;
    }

    assert(feature_num_ % 2 != 0); // features num: 2r + 1 

    write_real_dataset(datas, tags, get_cnts);
    // dataset_cnt_ += 1;
    return;
}

void ClfModel::make_train(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& tags, std::vector<uint32_t>& get_cnts) {
    assert(feature_num_ > 0);
    write_dataset(datas, tags, get_cnts);

    // already write dataset
    // send msg to LightGBM server, let server read dataset and train new model
    libsocket::inet_stream sock(host_, port_, LIBSOCKET_IPv4);
    std::string message = TRAIN_PREFIX + dataset_name_;
    // already write dataset, send dataset path to server
    // should not receive any message from server
    std::string recv_buffer;
    recv_buffer.resize(buffer_size_);
    sock << message;
    sock >> recv_buffer; // wait for training end
    // will destroy sock when leaving this func scope
}

void ClfModel::make_predict_samples(std::vector<std::vector<uint32_t>>& datas) {
    assert(feature_num_ > 0);
    csv2::Reader<csv2::delimiter<','>, 
                csv2::quote_character<'"'>, 
                csv2::first_row_is_header<true>,
                csv2::trim_policy::trim_whitespace> csv;
    std::vector<uint32_t> data;           
    if (csv.mmap(dataset_path_)) {
        // const auto header = csv.header();
        int cnt = 0;
        for (auto row : csv) {
            // only choose first 10 samples
            if ((++cnt) > 10) {
                break;
            }
            data.clear();
            for (auto cell : row) {
                std::string value;
                cell.read_value(value);
                data.emplace_back(stoul(value));
            }
            // remind that csv reader will read a empty row in the end, that is why !data.empty()
            // csv file last two column is real tag and get cnt
            // we need to pop out last column
            if (!data.empty()) {
                data.pop_back(); // pop out get cnt
                data.pop_back(); // pop out real tag
            }
            assert(data.size() == feature_num_);
            datas.emplace_back(data);
        }
    }
}

void ClfModel::make_real_predict(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& preds) {
    assert(preds.empty());
    libsocket::inet_stream sock(host_, port_, LIBSOCKET_IPv4);
    std::string message, recv_buffer;
    for (std::vector<uint32_t>& data : datas) {
        if (!data.empty()) {
            prepare_data(data);
            message.clear();
            recv_buffer.clear();
            recv_buffer.resize(buffer_size_);
            message = std::to_string(data[0]);
            for (size_t i = 1; i < data.size(); i ++) {
                message = message + " " + std::to_string(data[i]);
            }
            message = PREDICT_PREFIX + message;
            assert(message.size() <= buffer_size_);
            sock << message;
            // only receive pred tag integer
            sock >> recv_buffer;
            uint16_t pred = std::stoul(recv_buffer);
            assert(pred >= MIN_UNITS_NUM && pred <= MAX_UNITS_NUM);
            preds.emplace_back(pred);
        }
    }
    // only write pred result to vector preds, and return nothing
    assert(datas.size() == preds.size());
}

void ClfModel::make_predict(std::vector<std::vector<uint32_t>>& datas, std::vector<uint16_t>& preds) {
    preds.clear();

    // datas empty means we are debuging class ClfModel
    if (datas.empty()) {
        make_predict_samples(datas);
    } 
    // only write pred result to vector preds, and return nothing
    make_real_predict(datas, preds);
    return;
}

}