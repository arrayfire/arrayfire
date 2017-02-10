/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <dispatch.hpp>
#include <map>
#include <iostream>
#include <iomanip>
#include <string>
#include <types.hpp>
#include <err_opencl.hpp>

#include <MemoryManager.hpp>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

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

static MemoryManager &getMemoryManager()
{
    static MemoryManager instance;
    return instance;
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

static MemoryManagerPinned &getMemoryManagerPinned()
{
    static MemoryManagerPinned instance;
    return instance;
}

void setMemStepSize(size_t step_bytes)
{
    getMemoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize(void)
{
    return getMemoryManager().getMemStepSize();
}

size_t getMaxBytes()
{
    return getMemoryManager().getMaxBytes();
}

unsigned getMaxBuffers()
{
    return getMemoryManager().getMaxBuffers();
}

void garbageCollect()
{
    getMemoryManager().garbageCollect();
}

void printMemInfo(const char *msg, const int device)
{
    getMemoryManager().printInfo(msg, device);
}

template<typename T>
T* memAlloc(const size_t &elements)
{
    return (T *)getMemoryManager().alloc(elements * sizeof(T), false);
}

void* memAllocUser(const size_t &bytes)
{
    return getMemoryManager().alloc(bytes, true);
}
template<typename T>
void memFree(T *ptr)
{
    return getMemoryManager().unlock((void *)ptr, false);
}

void memFreeUser(void *ptr)
{
    getMemoryManager().unlock((void *)ptr, true);
}

cl::Buffer *bufferAlloc(const size_t &bytes)
{
    return (cl::Buffer *)getMemoryManager().alloc(bytes, false);
}

void bufferFree(cl::Buffer *buf)
{
    return getMemoryManager().unlock((void *)buf, false);
}

void memLock(const void *ptr)
{
    getMemoryManager().userLock((void *)ptr);
}

void memUnlock(const void *ptr)
{
    getMemoryManager().userUnlock((void *)ptr);
}

bool isLocked(const void *ptr)
{
    return getMemoryManager().isUserLocked((void *)ptr);
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes,  size_t *lock_buffers)
{
    getMemoryManager().bufferInfo(alloc_bytes, alloc_buffers,
                                  lock_bytes,  lock_buffers);
}

template<typename T>
T* pinnedAlloc(const size_t &elements)
{
    return (T *)getMemoryManagerPinned().alloc(elements * sizeof(T), false);
}

template<typename T>
void pinnedFree(T* ptr)
{
    return getMemoryManagerPinned().unlock((void *)ptr, false);
}

bool checkMemoryLimit()
{
    return getMemoryManager().checkMemoryLimit();
}

#define INSTANTIATE(T)                                      \
    template T* memAlloc(const size_t &elements);           \
    template void memFree(T* ptr);                          \
    template T* pinnedAlloc(const size_t &elements);        \
    template void pinnedFree(T* ptr);                       \

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(short)
    INSTANTIATE(ushort)
}
