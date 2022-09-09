//
// Created by joechen on 2022/4/3.
//

#pragma once

#include "rocksdb/rocksdb_namespace.h"

#include <deque>
#include <mutex>
#include <condition_variable>
#include "db/art/rwspinlock.h"

#include "heat_group.h"

namespace ROCKSDB_NAMESPACE {

/** @brief  A templated *thread-safe* collection based on dequeue
            pop_front() waits for the notification of a filling method if the collection is empty.
            The various "emplace" operations are factorized by using the generic "addData_protected".
            This generic asks for a concrete operation to use, which can be passed as a lambda.
**/
template< typename T >
class TQueueConcurrent {

  using const_iterator = typename std::deque<T>::const_iterator;

  using size_type = typename std::deque<T>::size_type;

 public:
  //! @brief Get size_ of the deque
  size_type size() {
    return _collection.size();
  }

  void clear() {
    std::unique_lock<std::mutex> lock{_mutex};
    _collection.clear();
  }

  //! @brief Emplaces a new instance of T in front of the deque

  template<typename... Args>
  void emplace_front( Args&&... args )
  {
    addData_protected( [&] {
      _collection.emplace_front(std::forward<Args>(args)...);
    } );
  }

  /** @brief Emplaces a new instance of T at the back of the deque **/
  template<typename... Args>
  void emplace_back( Args&&... args )
  {
    addData_protected( [&] {
      _collection.emplace_back(std::forward<Args>(args)...);
    } );
  }

  /** @brief  Returns the front element and removes it from the collection
              No exception is ever returned as we garanty that the deque is not empty
              before trying to return data.
  **/
  T pop_front( void ) noexcept
  {
    std::unique_lock<std::mutex> lock{_mutex};
    while (_collection.empty()) {
      _condNewData.wait(lock);
    }
    auto elem = std::move(_collection.front());
    _collection.pop_front();
    return elem;
  }

 private:

  /** @brief  Protects the deque, calls the provided function and notifies the presence of new data
      @param  The concrete operation to be used. It MUST be an operation which will add data to the deque,
              as it will notify that new data are available!
  **/
  template<class F>
  void addData_protected(F&& fct)
  {
    std::unique_lock<std::mutex> lock{ _mutex };
    fct();
    lock.unlock();
    _condNewData.notify_one();
  }

  std::deque<T> _collection;                     ///< Concrete, not thread safe, storage.

  std::mutex   _mutex;                    ///< Mutex protecting the concrete storage

  std::condition_variable _condNewData;   ///< Condition used to notify that new data are available.

};

template class TQueueConcurrent<GroupOperation>;

template class TQueueConcurrent<char *>;

class SegmentQueue {
 public:
  SegmentQueue(int max_count) {
    count_ = max_count * 2;
    queue_ = new int[max_count * 2];
    for (int i = 0; i < count_; ++i) {
      queue_[i] = -1;
      garbage_count_[i] = 0;
    }

    vptr_map = std::vector<std::unordered_map<uint32_t, uint64_t>>(1024);
    for (int i = 0; i < 1024; ++i) {
      vptr_map[i].clear();
    }
  }

  ~SegmentQueue() {
    delete[] queue_;
  }

  int GetSegment() {
    auto tail = tail_.load(std::memory_order_relaxed);
    int check_count = (tail - head_) / 16;

    int max_garbage_count = 0;
    int chosen_slot = -1;
    int chosen_segment = -1;

    for (int i = head_; i < head_ + check_count; ++i) {
      int slot = i % count_;
      int segment_id = queue_[slot];

      if (segment_id != -1 && garbage_count_[segment_id] > max_garbage_count) {
        chosen_slot = slot;
        chosen_segment = segment_id;
        max_garbage_count =
            garbage_count_[segment_id].load(std::memory_order_relaxed);
      }
    }

    // prune
    while (queue_[head_ % count_] == -1) {
      ++head_;
    }

    if (unlikely(chosen_slot == -1)) {
      chosen_slot = (head_++) % count_;
      chosen_segment = queue_[chosen_slot];
    }

    assert(chosen_segment >= 0);
    queue_[chosen_slot] = -1;
    return chosen_segment;
  }

  void PushSegment(int segment_id) {
    int cur_tail = tail_.load(std::memory_order_relaxed);
    while (!tail_.compare_exchange_strong(cur_tail, cur_tail + 1));
    queue_[(cur_tail - 1) % count_] = segment_id;
  }

  void ClearGarbage(int segment_id) {
    garbage_count_[segment_id] = 0;
  }

  void AddGarbage(uint32_t hash, uint64_t vptr) {
    auto bucket_id = hash & 1023;
    std::lock_guard<RWSpinLock> lk(map_locks[bucket_id]);
    auto& map = vptr_map[bucket_id];
    auto iter = map.find(hash);
    if (unlikely(iter == map.end())) {
      map[hash] = vptr;
      return;
    }

    auto old_vptr = iter->second;
    map[hash] = vptr;
    ++garbage_count_[old_vptr >> 20];
  }

 private:
  int head_ = 0;

  std::atomic<int> tail_{1};

  int count_;

  int* queue_;

  std::atomic<int> garbage_count_[8192];

  std::vector<std::unordered_map<uint32_t, uint64_t>> vptr_map;

  RWSpinLock map_locks[1024];

};

}  // namespace ROCKSDB_NAMESPACE
