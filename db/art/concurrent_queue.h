//
// Created by joechen on 2022/4/3.
//

#pragma once

#include "rocksdb/rocksdb_namespace.h"

#include <deque>
#include <mutex>
#include <condition_variable>

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

}  // namespace ROCKSDB_NAMESPACE
