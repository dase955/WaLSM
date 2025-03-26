#include <csv2/writer.hpp>
#include <csv2/reader.hpp>
#include <libsocket/exception.hpp>
#include <libsocket/inetclientstream.hpp>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <thread>

// you need to modify features num in lgb_server and filter cache!!!

void write_debug_dataset(std::string& path) {
    // ready for writer
    std::ofstream stream(path);
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
        header.emplace_back("Range_" + std::to_string(i));
        header.emplace_back("Hotness_" + std::to_string(i));
    }
    header.emplace_back("Target");
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
        uint32_t target = 5 - level;
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if (r > 0.10 * level) {
            target -= 1;
        }

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
        values.emplace_back(std::to_string(level));
        for (int j = 0; j < 20; j ++) {
            values.emplace_back(std::to_string(ids[j]));
            values.emplace_back(std::to_string(uint32_t(1e6 * hotness_map[ids[j]])));
        }
        values.emplace_back(std::to_string(target));

        rows.emplace_back(values);
    }

    writer.write_rows(rows);
    stream.close();
}

void make_predict_samples(std::string& path, std::vector<std::vector<uint32_t>>& datas) {
    datas.clear();
    csv2::Reader<csv2::delimiter<','>, 
                csv2::quote_character<'"'>, 
                csv2::first_row_is_header<true>,
                csv2::trim_policy::trim_whitespace> csv;
               
    if (csv.mmap(path)) {
        const auto header = csv.header();
        // int cnt = 0;
        for (auto row : csv) {
            
            /*
            if ((++cnt) > 10) {
                break;
            }
            */
            
            // cnt ++;
            std::vector<uint32_t> data;
            for (auto cell : row) {
                std::string value;
                cell.read_value(value);
                data.emplace_back(stoul(value));
            }
            if (!data.empty()) {
                data.pop_back();
            }
            datas.emplace_back(data);
        }
    }
}

void build_message(std::vector<uint32_t>& data, std::string& message) {
    message.clear();
    message = std::to_string(data[0]);
    for (size_t i = 1; i < data.size(); i ++) {
        message = message + " " + std::to_string(data[i]);
    }
}

int main() {
    std::string host = "127.0.0.1";
    std::string port = "9090";
    std::string recv;
    size_t recv_size = 1024;
    std::string file = "dataset.csv";

    recv.resize(recv_size);

    write_debug_dataset(file);

    libsocket::inet_stream t1_sock(host, port, LIBSOCKET_IPv4);
    std::string msg = "t " + file;
    // already write dataset, send dataset path to server
    // should not receive any message from server
    t1_sock << msg;

    // t1_sock.shutdown();

    std::vector<std::vector<uint32_t>> datas;
    
    make_predict_samples(file, datas);
    libsocket::inet_stream p1_sock(host, port, LIBSOCKET_IPv4);
    for (std::vector<uint32_t>& data : datas) {
        if (!data.empty()) {
            build_message(data, msg);
            msg = "p " + msg;
            p1_sock << msg;
            p1_sock >> recv;
            // train model need enough time, so should always receive 5 (default class)
            std::cout << "receive " << recv.size() <<  " bytes : " << recv << std::endl;
        }
    }

    constexpr int sleep_time = 10000;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    libsocket::inet_stream p2_sock(host, port, LIBSOCKET_IPv4);
    for (std::vector<uint32_t>& data : datas) {
        if (!data.empty()) {
            build_message(data, msg);
            msg = "p " + msg;
            p2_sock << msg;
            p2_sock >> recv;
            // already train model, receive data class predicted by the model
            std::cout << "receive " << recv.size() <<  " bytes : " << recv << std::endl;
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    write_debug_dataset(file);
    libsocket::inet_stream t2_sock(host, port, LIBSOCKET_IPv4);
    msg = "t " + file;
    // already write dataset, send dataset path to server
    // should not re
    t2_sock << msg;

    return 0;
}