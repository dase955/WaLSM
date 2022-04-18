//
// Created by joechen on 22-4-18.
//

#include <cstdint>
#include <string>
#include <unordered_map>
#include <rocksdb/rocksdb_namespace.h>

namespace ROCKSDB_NAMESPACE {

void InitializeMemory(std::unordered_map<std::string, int64_t>& memory_usages);

char* GetMappedAddress(const std::string& name);

void UnmapMemory();

} // namespace ROCKSDB_NAMESPACE
