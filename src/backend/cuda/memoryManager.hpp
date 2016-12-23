/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <platform.hpp>
#include <MemoryManager.hpp>

namespace cuda
{

class MemoryManager  : public common::MemoryManager
{
    int getActiveDeviceId();
    size_t getMaxMemorySize(int id);
public:
    MemoryManager();
    void *nativeAlloc(const size_t bytes);
    void nativeFree(void *ptr);
    ~MemoryManager()
    {
        common::lock_guard_t lock(this->memory_mutex);
        for (int n = 0; n < getDeviceCount(); n++) {
            try {
                cuda::setDevice(n);
                this->garbageCollect();
            } catch(AfError err) {
                continue; // Do not throw any errors while shutting down
            }
        }
    }
};

// CUDA Pinned Memory does not depend on device
// So we pass 1 as numDevices to the constructor so that it creates 1 vector
// of memory_info
// When allocating and freeing, it doesn't really matter which device is active
class MemoryManagerPinned  : public common::MemoryManager
{
    int getActiveDeviceId();
    size_t getMaxMemorySize(int id);
public:
    MemoryManagerPinned();
    void *nativeAlloc(const size_t bytes);
    void nativeFree(void *ptr);
    ~MemoryManagerPinned()
    {
        common::lock_guard_t lock(this->memory_mutex);
        this->garbageCollect();
    }
};

}
