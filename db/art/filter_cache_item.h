#pragma once

#include <iostream>
#include <fstream>
#include <mutex>
#include <cassert>
#include <vector>
#include <map>
#include "macros.h"

namespace ROCKSDB_NAMESPACE {

// 先在filter cache里为每个segment默认启用总bits-per-key=8，随着写入的segment的增加，
// 一旦已经占用了filter cache最大容量的一定阈值(如80%), 就利用GreedyAlgo计算规划问题，并进行模型训练
// 一旦filter cache已满，就进入filter cache的double heap调整，我们只需将新的segment用模型进行预测
// 将新segment的node插入到两个heap里，在后台启动一个线程，自行调整两个堆，并不断返回调整的结果
// 得到结果后，我们可以立即对filter units的启用情况进行调节，也可以先保存后面批量调整
// 具体见文档


// 注意加上一些必要的英文注释
// filter cache主要为一个map, key是segment id(uint32_t), value就为FilterCacheItem类
// 成员函数需要在filter_cache_item.cc里定义
class FilterCacheItem {
private:
    // 这里定义一些必要的成员变量，尽量设置为private
    // 可以存handle、segment id等信息
    // 允许使用STL类，如vector、map等
    // 是否需要使用mutex来保证filter units的启用/禁用管理与用units检查key二者不冲突?
public:
    // 构造函数，可以初始化成员变量
    FilterCacheItem(const uint32_t& segment_id);

    // 清理成员变量，避免内存泄漏，如果new了空间，就可能需要在这里清理
    ~FilterCacheItem();

    // 占用的内存空间，这里估计总共使用的filter units占用的空间就行了
    // 注意，返回的空间大小为占用的bits数量，不是bytes数量
    uint32_t approximate_size();

    // 根据目前已经启用的units数，启用或禁用filter units
    // 输入需要启用的units数，决定启用、禁用还是不处理
    // units_num : [MIN_UNITS_NUM, MAX_UNITS_NUM]
    void enable_units(const uint32_t& units_num);

    // 输入一个key，判断是否存在
    // 具体就是从第一个unit开始，依次判断，如果有一个unit判断不存在。就停止。
    // 如果每个unit都判断存在，就返回true，否则返回false
    // 如果启用的unit数为0，默认返回true
    bool check_key(const std::string& key);
};

}