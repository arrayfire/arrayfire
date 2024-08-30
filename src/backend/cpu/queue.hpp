/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Param.hpp>
#include <common/util.hpp>
#include <memory.hpp>

#include <algorithm>

// FIXME: Is there a better way to check for std::future not being supported ?
#if defined(AF_DISABLE_CPU_ASYNC) || \
    (defined(__GNUC__) &&            \
     (__GCC_ATOMIC_INT_LOCK_FREE < 2 || __GCC_ATOMIC_POINTER_LOCK_FREE < 2))

#include <functional>
using std::function;
#include <err_cpu.hpp>
#define __SYNCHRONOUS_ARCH 1
class queue_impl {
   public:
    template<typename F, typename... Args>
    void enqueue(const F func, Args... args) const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
    }

    void sync() const { AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL); }

    bool is_worker() const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return false;
    }
};

class event_impl {
   public:
    event_impl() noexcept                              = default;
    ~event_impl() noexcept                             = default;
    explicit event_impl(const event_impl &other)       = default;
    event_impl(event_impl &&other) noexcept            = default;
    event_impl &operator=(event_impl &&other) noexcept = default;
    event_impl &operator=(event_impl &other) noexcept  = default;

    explicit event_impl(const int val) {}

    event_impl &operator=(int val) noexcept { return *this; }

    int create() {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return 0;
    }

    int mark(queue_impl &queue) {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return 0;
    }

    int wait(queue_impl &queue) const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return 0;
    }

    int sync() const noexcept {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return 0;
    }

    operator bool() const noexcept { return false; }
};

#else

#include <threads/async_queue.hpp>
#include <threads/event.hpp>
#define __SYNCHRONOUS_ARCH 0
using queue_impl = threads::async_queue;
using event_impl = threads::event;

#endif

namespace arrayfire {
namespace cpu {

/// Wraps the async_queue class
class queue {
   public:
    queue()
        : count(0)
        , sync_calls(__SYNCHRONOUS_ARCH == 1 ||
                     common::getEnvVar("AF_SYNCHRONOUS_CALLS") == "1") {}

    template<typename F, typename... Args>
    void enqueue(const F func, Args &&...args) {
        count++;
        if (sync_calls) {
            func(toParam(std::forward<Args>(args))...);
        } else {
            aQueue.enqueue(func, toParam(std::forward<Args>(args))...);
        }
#ifndef NDEBUG
        sync();
#else
        if (getMemoryPressure() >= getMemoryPressureThreshold() ||
            count >= 25) {
            sync();
        }
#endif
    }

    void sync() {
        count = 0;
        if (!sync_calls) aQueue.sync();
    }

    bool is_worker() const {
        return (!sync_calls) ? aQueue.is_worker() : false;
    }

    friend class queue_event;

   private:
    int count;
    const bool sync_calls;
    queue_impl aQueue;
};

class queue_event {
    event_impl event_;

   public:
    queue_event() = default;
    queue_event(int val) : event_(val) {}

    int create() { return event_.create(); }

    int mark(queue &q) { return event_.mark(q.aQueue); }
    int wait(queue &q) { return event_.wait(q.aQueue); }
    int sync() noexcept { return event_.sync(); }
    operator bool() const noexcept { return event_; }
};
}  // namespace cpu
}  // namespace arrayfire
