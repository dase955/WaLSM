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

}  // namespace ROCKSDB_NAMESPACE