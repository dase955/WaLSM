//
// Created by joechen on 2022/4/3.
//

#pragma once

#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

#ifndef ART_LITTLE_ENDIAN
#define ART_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
#endif

/*
 * Macros for global_memtable.h
 */

#define LAST_CHAR 255

#define NVM_MAX_ROWS              14
#define NVM_MAX_SIZE              224

// Macro for InnerNode status_
#define IS_LEAF(s)                ((s) & 0x80000000)
#define SET_LEAF(s)               (s) |= 0x80000000
#define SET_NON_LEAF(s)           (s) &= ~(0x80000000)

#define IS_ART_FULL(s)            ((s) & 0x40000000)
#define SET_ART_FULL(s)           (s) |= 0x40000000
#define SET_ART_NON_FULL(s)       (s) &= ~(0x40000000)

#define IS_GROUP_START(s)         ((s) & 0x20000000)
#define SET_GROUP_START(s)        (s) |= 0x20000000
#define SET_NON_GROUP_START(s)    (s) &= ~(0x20000000)

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

#define GET_SIZE(hdr)           GET_VALUE((hdr), 6)
#define SET_SIZE(hdr, val)      SET_VALUE((hdr), 6, (val))

#define GET_ROWS(hdr)           GET_VALUE((hdr), 5)
#define SET_ROWS(hdr, val)      SET_VALUE((hdr), 5, (val))

#define GET_PRELEN(hdr)         GET_VALUE((hdr), 4)
#define SET_PRELEN(hdr, val)    SET_VALUE((hdr), 4, (val))

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

#define ALT_FIRST_TAG           0x8000000000000000
#define VALID_TAG               0x2000000000000000

/*
 * Macros for HeatGroup, Compaction and TimeStamps
 */

#define MAX_LAYERS 10

// BASE_LAYER is for groups whose size_ is below threshold.
#define BASE_LAYER (-1)

// TEMP_LAYER is for groups doing compaction.
#define TEMP_LAYER (-2)

}  // namespace ROCKSDB_NAMESPACE