//
// Created by joechen on 2022/4/3.
//

#pragma once

#include <rocksdb/rocksdb_namespace.h>

#ifdef USE_PMEM
#include <libpmem.h>
#endif

namespace ROCKSDB_NAMESPACE {

#ifndef ART_LITTLE_ENDIAN
#define ART_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
#endif

#define ALIGN_UP(x, align)   (((x) + ((align) - 1)) & ~((align) - 1))
#define ALIGN_DOWN(x, align) ((x) & ~((align) - 1))
#define SIZE_TO_ROWS(size)   ((size) >> 4)
#define SIZE_TO_BYTES(size)  ((size) << 4)
#define ROW_TO_SIZE(rows)    ((rows) << 4)

/*
 * Macros for global_memtable.h
 */

#define LAST_CHAR            255
#define ROW_BYTES            256
#define ROW_SIZE             16
#define NVM_MAX_ROWS         14
#define NVM_MAX_SIZE         224
#define BULK_WRITE_SIZE      208

#define INITIAL_STATUS(s)         (0x80000000 | (s))

// Macro for InnerNode status_
#define IS_LEAF(n)                ((n)->status_ & 0x80000000)
#define NOT_LEAF(n)               (!IS_LEAF(n))
#define SET_LEAF(n)               ((n)->status_) |= 0x80000000
#define SET_NON_LEAF(n)           ((n)->status_) &= ~(0x80000000)

#define IS_ART_FULL(n)            (((n)->status_) & 0x40000000)
#define SET_ART_FULL(n)           ((n)->status_) |= 0x40000000
#define SET_ART_NON_FULL(n)       ((n)->status_) &= ~(0x40000000)

#define IS_GROUP_START(n)         (((n)->status_) & 0x20000000)
#define NOT_GROUP_START(n)         (!IS_GROUP_START(n))
#define SET_GROUP_START(n)        ((n)->status_) |= 0x20000000
#define SET_NON_GROUP_START(n)    ((n)->status_) &= ~(0x20000000)

#define IS_INVALID(s)             ((s) & 0x10000000)
#define SET_NODE_INVALID(s)       (s) |= 0x10000000

#define GET_GC_FLUSH_SIZE(s)        (((s) >> 16) & 0x000000ff)
#define SET_GC_FLUSH_SIZE(s, size)  (s) &= 0xff00ffff; (s) |= ((size) << 16)

#define GET_NODE_BUFFER_SIZE(s)        ((s) & 0x0000ffff)
#define SET_NODE_BUFFER_SIZE(s, size)  (s) &= 0xffff0000; (s) |= (size)

/*
 * Macros for nvm_Node.h
 */

#define PAGE_SIZE (4096)

#define CLEAR_VALUE(hdr, pos) hdr &= ~((uint64_t)0xff << ((pos) << 3));
#define GET_VALUE(hdr, pos)   (((hdr) >> ((pos) << 3)) & 0xff)
#define SET_VALUE(hdr, pos, val) \
    CLEAR_VALUE(hdr, pos)       \
    (hdr) |= ((uint64_t)(val) << ((pos) << 3))

// Currently, size is actually unused
#define GET_SIZE(hdr)           GET_VALUE((hdr), 6)
#define SET_SIZE(hdr, val)      SET_VALUE((hdr), 6, (val))

#define GET_ROWS(hdr)           GET_VALUE((hdr), 5)
#define SET_ROWS(hdr, val)      SET_VALUE((hdr), 5, (val))

#define GET_LEVEL(hdr)          GET_VALUE((hdr), 4)
#define SET_LEVEL(hdr, val)     SET_VALUE((hdr), 4, (val))

#ifdef ART_LITTLE_ENDIAN
#define SET_LAST_PREFIX(hdr, c) ((uint8_t *)&(hdr))[0] = (c)
#define GET_LAST_PREFIX(hdr)    ((uint8_t *)&(hdr))[0]
#else
#define SET_LAST_PREFIX(hdr, c) ((uint8_t *)&(hdr))[7] = (c)
#define GET_LAST_PREFIX(hdr)    ((uint8_t *)&(hdr))[7]
#endif

#define CLEAR_TAG(hdr, tag)     (hdr) &= ~(tag)
#define GET_TAG(hdr, tag)       ((hdr) & (tag))
#define SET_TAG(hdr, tag)       (hdr) |= (tag)
#define SET_NVM_TAG(node, tag)  SET_TAG((node)->meta.header, tag)

// Alt bit maybe redundant
#define ALT_FIRST_TAG           0x8000000000000000
#define GROUP_START_TAG         0x4000000000000000
#define VALID_TAG               0x2000000000000000
#define DUMMY_TAG               0x1000000000000000

/*
 * Macros for HeatGroup, Compaction and TimeStamps
 */

#define MAX_LAYERS 10

// BASE_LAYER is for groups whose size is below threshold.
#define BASE_LAYER (-1)

// TEMP_LAYER is for groups doing compaction.
#define TEMP_LAYER (-2)

/////////////////////////////////////////////////////
// PMem

#define MEMORY_BARRIER __asm__ volatile("mfence":::"memory")

#ifndef USE_PMEM
#define MEMCPY(des, src, size, flag) memcpy((des), (src), (size))
#define PERSIST(ptr, len)
#define FLUSH(addr, len)
#define NVM_BARRIER
#define CLWB(ptr, len)
#else
#define MEMCPY(des, src, size, flags) \
  pmem_memcpy((des), (src), (size), flags)
// PERSIST = FLUSH + FENCE
#define PERSIST(addr, len) pmem_persist((addr), (len))
#define FLUSH(addr, len) pmem_flush(addr, len)
#define NVM_BARRIER pmem_drain()
#define CLWB(ptr, len)
#endif

/*
 * Macros for additional work on WaLSM -- WaLSM+
 */

// micros for HeatBuckets

// hotness update formula
#define BUCKETS_ALPHA 0.2  
// samples pool max size, using reservoir sampling
#define SAMPLES_LIMIT 10000 
// if recv samples exceed SAMPLES_MAXCNT, end reservoir sampling and init Heat Buckets
#define SAMPLES_MAXCNT 5000000 
// short period get count, if get count equal to or exceed PERIOD_COUNT, 
// end this short period and start next short period
#define PERIOD_COUNT 50000 
// number of heat buckets (number of key ranges, see hotness estimating in the paper)
#define DEFAULT_BUCKETS_NUM 500 
// magic number in class HeatBuckets
#define MAGIC_FACTOR 500 

// micros for Model Train

// long period = TRAIN_PERIODS * short period. if one long period end, evaluate model and retrain model if necessary
#define TRAIN_PERIODS 10 
// dataset csv file name
#define DATASET_NAME "dataset.csv"
// the path to save model txt file and train dataset csv file
#define MODEL_PATH "/pg_wal/ycc/" 
// we cannot send hotness value (double) to model side, 
// so we try multiple hotness value by HOTNESS_SIGNIFICANT_DIGITS_FACTOR, then send its integer part to model
// also we need to multiple key range rate by RATE_SIGNIFICANT_DIGITS_FACTOR
#define HOTNESS_SIGNIFICANT_DIGITS_FACTOR 1e6 
#define RATE_SIGNIFICANT_DIGITS_FACTOR 1e3
// model feature num max limit : 2 * 45 + 1
#define MAX_FEATURES_NUM 91

// config micro connecting to LightGBM server 

// we use Inet socket to connect server
#define HOST "127.0.0.1"
#define PORT "9090"
// max size of socket receive buffer size
#define BUFFER_SIZE 1024
// socket message prefix
#define TRAIN_PREFIX "t "
#define PREDICT_PREFIX "p "

// micros for filter cache

// before model work, we enable DEFAULT_UNITS_NUM units for every segments
#define DEFAULT_UNITS_NUM 4
// bits-per-key for every filter unit of every segment, 
// found default bits-per-key = DEFAULT_UNITS_NUM * BITS_PER_KEY_PER_UNIT = 10
// equal to primary value of paper benchmark config value
#define BITS_PER_KEY_PER_UNIT 4
// max unit nums for every segment, we only generate MAX_UNITS_NUM units for every segment
#define MAX_UNITS_NUM 8
// we enable 0 unit for coldest segments
#define MIN_UNITS_NUM 0
// default max size of cache space : 8 * 1024 * 1024 * 128 = 1073741824 bit = 128 MB
#define CACHE_SPACE_SIZE 1073741824
// fitler cache helper heap type
#define BENEFIT_HEAP 0
#define COST_HEAP 1
#define UNKNOWN_HEAP 2
// visit cnt update bound
#define VISIT_CNT_UPDATE_BOUND 10
// filter cache map threshold
#define FULL_RATE 0.95
#define READY_RATE 0.60
// default init L0 counts
#define INIT_LEVEL_0_COUNT 0
// default size of one filter unit (bits)
#define DEFAULT_UNIT_SIZE 0
// inherit remain factor
#define INHERIT_REMAIN_FACTOR 0.5

// filter cache client background threads num
#define FILTER_CACHE_THREADS_NUM 10

}  // namespace ROCKSDB_NAMESPACE