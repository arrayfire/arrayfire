/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memoryManager.hpp>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

namespace opencl
{

int MemoryManager::getActiveDeviceId()
{
    return opencl::getActiveDeviceId();
}

size_t MemoryManager::getMaxMemorySize(int id)
{
    return opencl::getDeviceMemorySize(id);
}

MemoryManager::MemoryManager() :
    common::MemoryManager(getDeviceCount(), common::MAX_BUFFERS, AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

void *MemoryManager::nativeAlloc(const size_t bytes)
{
    return (void *)(new cl::Buffer(getContext(), CL_MEM_READ_WRITE, bytes));
}

void MemoryManager::nativeFree(void *ptr)
{
    delete (cl::Buffer *)ptr;
}

int MemoryManagerPinned::getActiveDeviceId()
{
    return opencl::getActiveDeviceId();
}

size_t MemoryManagerPinned::getMaxMemorySize(int id)
{
    return opencl::getDeviceMemorySize(id);
}

MemoryManagerPinned::MemoryManagerPinned() :
    common::MemoryManager(getDeviceCount(), common::MAX_BUFFERS, AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG),
    pinned_maps(getDeviceCount())
{
    this->setMaxMemorySize();
}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes)
{
    void *ptr = NULL;
    cl::Buffer buf= cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, bytes);
    ptr = getQueue().enqueueMapBuffer(buf, true, CL_MAP_READ | CL_MAP_WRITE, 0, bytes);
    pinned_maps[opencl::getActiveDeviceId()][ptr] = buf;
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr)
{
    int n = opencl::getActiveDeviceId();
    auto iter = pinned_maps[n].find(ptr);

    if (iter != pinned_maps[n].end()) {
        getQueue().enqueueUnmapMemObject(pinned_maps[n][ptr], ptr);
        pinned_maps[n].erase(iter);
    }
}

}
