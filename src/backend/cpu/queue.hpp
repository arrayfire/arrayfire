/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <util.hpp>
#include <memory.hpp>
#include <Param.hpp>

//FIXME: Is there a better way to check for std::future not being supported ?
#if defined(AF_DISABLE_CPU_ASYNC) || (defined(__GNUC__) && (__GCC_ATOMIC_INT_LOCK_FREE < 2 || __GCC_ATOMIC_POINTER_LOCK_FREE < 2))

#include <functional>
using std::function;
#include <err_cpu.hpp>
#define __SYNCHRONOUS_ARCH 1
class queue_impl
{
public:
    template <typename F, typename... Args>
    void enqueue(const F func, Args... args) const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
    }

    void sync() const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
    }

    bool is_worker() const {
        AF_ERROR("Incorrectly configured", AF_ERR_INTERNAL);
        return false;
    }

};

#else

#include <async_queue.hpp>
#define __SYNCHRONOUS_ARCH 0
typedef async_queue queue_impl;

#endif

#pragma once

namespace cpu {

/// Wraps the async_queue class
class queue
{
public:
    queue()
        :
        count(0),
        sync_calls( __SYNCHRONOUS_ARCH == 1 || getEnvVar("AF_SYNCHRONOUS_CALLS") == "1")
    {}

    template <typename F, typename... Args>
    void enqueue(const F func, Args... args)
    {
        count++;
        if(sync_calls) { func(toParam(args)... ); }
        else           { aQueue.enqueue(func, toParam(args)... ); }
#ifndef NDEBUG
        sync();
#else
        if (checkMemoryLimit() || count >= 25) {
            sync();
        }
#endif
    }

    void sync()
    {
        count = 0;
        if(!sync_calls) aQueue.sync();
    }

    bool is_worker() const
    {
        return (!sync_calls) ? aQueue.is_worker() : false;
    }

    private:
        int count;
        const bool sync_calls;
        queue_impl aQueue;
};

}
