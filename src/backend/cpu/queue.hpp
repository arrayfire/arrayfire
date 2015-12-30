/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <util.hpp>
#include <async_queue.hpp>

#pragma once

namespace cpu {

/// Wraps the async_queue class
class queue {
public:
  queue()
    : sync_calls( getEnvVar("AF_SYNCHRONOUS_CALLS") == "1") {}

  template <typename F, typename... Args>
  void enqueue(const F func, Args... args) {

    if(sync_calls) { func( args... ); }
    else           { aQueue.enqueue( func, args... ); }
#ifndef NDEBUG
    sync();
#endif

  }
  void sync() {
    if(!sync_calls) aQueue.sync();
  }

  bool is_worker() const {
    return (!sync_calls) ? aQueue.is_worker() : false;
  }

private:
  const bool sync_calls;
  async_queue aQueue;
};

}
