//
// Created by Guo Teng.
//

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>

namespace ROCKSDB_NAMESPACE {

// record evaluation metric for WaLSM paper
constexpr bool EVALUATE_METRIC = true; 

// 将FlushMetric和SSDWriteMetric加起来才是总的Write Data Size

// metric of WaLSM flush write evaluation
// NVM MemTable-L0的写入
// struct NVMWriteMetric {
//     uint64_t WriteSsdDataBytes = 0;
//     uint64_t LastPrintedBytes = 0;
  
//     std::string getMetric() {
//         double WriteSSdDataGB = WriteSsdDataBytes * 1.0 / (1024 * 1024 * 1024);
//         std::string s = "Write Metric analysis:";
//         s = s + " <WriteSSD> From NVM - " + std::to_string(WriteSSdDataGB) + " GB";
//         return s;
//     }
  
//     void printMetric() {
//         if (WriteSsdDataBytes - LastPrintedBytes > 1024 * 1024 * 1024) {
//             LastPrintedBytes = WriteSsdDataBytes;
//             std::cout << getMetric() << std::endl;
//         }
//     }

//     void updateMetric(uint64_t add_bytes) {
//         if (EVALUATE_METRIC) {
//             WriteSsdDataBytes += add_bytes;
//             printMetric();
//         }
//     }
// };

// metric of WaLSM compaction write evaluation
// L0-L1, L1-L2, ... 的Compaction的写入
struct SSDWriteMetric {
    uint64_t WriteSsdDataBytes = 0;
    uint64_t LastPrintedBytes = 0;
  
    std::string getMetric() {
        double WriteSSdDataGB = WriteSsdDataBytes * 1.0 / (1024 * 1024 * 1024);
        std::string s = "Write Metric analysis:";
        s = s + " <WriteSSD> From SSD - " + std::to_string(WriteSSdDataGB) + " GB";
        return s;
    }
  
    void printMetric() {
        if (WriteSsdDataBytes - LastPrintedBytes > 1024 * 1024 * 1024) {
            LastPrintedBytes = WriteSsdDataBytes;
            std::cout << getMetric() << std::endl;
        }
    }

    void updateMetric(uint64_t add_bytes) {
        if (EVALUATE_METRIC) {
            WriteSsdDataBytes += add_bytes;
            printMetric();
        }
    }
};

// metric of WaLSM flush evaluation
// 从NVM flush到SSD的KV数据大小
struct FlushMetric {
    uint64_t FlushSsdDataBytes = 0;
    uint64_t LastPrintedBytes = 0;
  
    std::string getMetric() {
        double FlushSSdDataGB = FlushSsdDataBytes * 1.0 / (1024 * 1024 * 1024);
        std::string s = "Flush Metric analysis:";
        s = s + " <FlushSSD> - " + std::to_string(FlushSSdDataGB) + " GB";
        return s;
    }
  
    void printMetric() {
        if (FlushSsdDataBytes - LastPrintedBytes > 512 * 1024 * 1024) {
            LastPrintedBytes = FlushSsdDataBytes;
            std::cout << getMetric() << std::endl;
        }
    }

    void updateMetric(uint64_t add_bytes) {
        if (EVALUATE_METRIC) {
            FlushSsdDataBytes += add_bytes;
            printMetric();
        }
    }
};
  
// metric of WaLSM read evaluation
// 读取的文件物理块个数，一个块4KB
struct ReadMetric {
    uint64_t ReadSsdBlocksCnt = 0;
    uint64_t LastPrintedCount = 0;
  
    std::string getMetric() {
        std::string s = "Read Metric analysis:";
        s = s + " <ReadSSD> - " + std::to_string(ReadSsdBlocksCnt) + " Blocks";
        return s;
    }
  
    void printMetric() {
        if (ReadSsdBlocksCnt - LastPrintedCount > 1000 * 1000) {
            LastPrintedCount = ReadSsdBlocksCnt;
            std::cout << getMetric() << std::endl;
        }
    }

    void updateMetric(uint64_t offset_start, uint64_t offset_end) {
        uint64_t block_start = offset_start / 4096;
        uint64_t block_end = offset_end / 4096;
        
        if (EVALUATE_METRIC) {
            ReadSsdBlocksCnt += std::max(block_end - block_end, uint64_t(1));
            printMetric();
        }
    }
};


} // namespace rocksdb