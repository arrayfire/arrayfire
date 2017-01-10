/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>
#include <MemoryManager.hpp>

#include <map>

namespace opencl
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
            opencl::setDevice(n);
            this->garbageCollect();
        }
    }
};

class MemoryManagerPinned  : public common::MemoryManager
{
    std::vector<
        std::map<void *, cl::Buffer>
        > pinned_maps;
    int getActiveDeviceId();
    size_t getMaxMemorySize(int id);

public:

    MemoryManagerPinned();

    void *nativeAlloc(const size_t bytes);
    void nativeFree(void *ptr);

    ~MemoryManagerPinned()
    {
        common::lock_guard_t lock(this->memory_mutex);
        for (int n = 0; n < getDeviceCount(); n++) {
            opencl::setDevice(n);
            this->garbageCollect();
            auto pinned_curr_iter = pinned_maps[n].begin();
            auto pinned_end_iter  = pinned_maps[n].end();
            while (pinned_curr_iter != pinned_end_iter) {
                pinned_maps[n].erase(pinned_curr_iter++);
            }
        }
    }
};

}
