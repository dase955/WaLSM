#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <set>
#include <cassert>

int main() {
    // simple args, avoid using command line arg or config file
    const std::string input_workload = "workload";
    const std::string output_seperators = "seperators";
    const int key_range_num = 500;

    std::set<std::string> keys;
    std::string key;

    std::ifstream in;
    in.open(input_workload.c_str(), std::ios::in);

    if (!in.is_open()) {
        std::cout << "failed to open input file, please check whether the file exists" << std::endl;
        return 0;
    }

    std::cout << "success open input file" << std::endl;
    while (std::getline(in, key)) {
        // use set STL to deduplicate keys
        keys.insert(key);
    } 
    std::cout << "unique key count : " << keys.size() << std::endl;
    in.close();

    // convert to vector in order to locate key range border
    std::vector<std::string> selector;
    selector.assign(keys.begin(), keys.end());
    std::cout << "selector size : " << selector.size() << std::endl;

    const int keys_size = selector.size();
    int one_key_range_count = 0;
    // remind that last key range may have more than (one_key_range_count) keys, but it doesnt matter
    one_key_range_count = keys_size / key_range_num;

    std::cout << "one key range count : " << one_key_range_count << std::endl;
    std::cout << "key range num : " << key_range_num << std::endl;

    // each key range : [k_m, k_M), we need locate (key_range_num-1) seperators - k1, k2, ...
    // when one key k comes, we can get key range index i, making (k >= seperators[i] && k < seperators[i+1])
    // we introduce two guard border, k_min = "u" and k_max = keys[keys.size()-1] + "MAXGUARD"
    // note k_min and k_max need modified according to your unique workload
    std::vector<std::string> seperators;
    seperators.push_back("user"); // min guard
    assert(seperators[0] < selector[0]);
    for (int i=1; i <= key_range_num-1; i++) { // need (key_range_num-1) seperators
        seperators.push_back(selector[i * one_key_range_count]);
    }
    seperators.push_back(selector[selector.size()-1]+"MAXGUARD"); // max guard

    std::ofstream out;
    out.open(output_seperators.c_str(), std::ios::out);

    if (!out.is_open()) {
        std::cout << "failed to open output file, please rerun to create output" << std::endl;
        return 0;
    }
    std::cout << "success open output file" << std::endl;

    for (const auto &seperator : seperators) {
        // std::cout << seperator << std::endl;
        out << seperator << std::endl;
    }
    out.close();

    /*
    std::ifstream valid;
    valid.open("seperators", std::ios::in);

    while (std::getline(valid, key)) {
        std::cout << key << std::endl;
    } 
    valid.close();
    */

    return 0;
}