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

MemoryManager::MemoryManager() :
    common::MemoryManager(getDeviceCount(), MAX_BUFFERS, MAX_BYTES, AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG)
{}

void *MemoryManager::nativeAlloc(const size_t bytes)
{
    try {
        return (void *)(new cl::Buffer(getContext(), CL_MEM_READ_WRITE, bytes));
    } catch(cl::Error err) {
        CL_TO_AF_ERROR(err);
    }
}

void MemoryManager::nativeFree(void *ptr)
{
    try {
        delete (cl::Buffer *)ptr;
    } catch(cl::Error err) {
        CL_TO_AF_ERROR(err);
    }
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

MemoryManagerPinned::MemoryManagerPinned() :
    common::MemoryManager(getDeviceCount(), MAX_BUFFERS, MAX_BYTES, AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG),
    pinned_maps(getDeviceCount())
{}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes)
{
    void *ptr = NULL;
    try {
        cl::Buffer buf= cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, bytes);
        ptr = getQueue().enqueueMapBuffer(buf, true, CL_MAP_READ | CL_MAP_WRITE, 0, bytes);
        pinned_maps[opencl::getActiveDeviceId()][ptr] = buf;
    } catch(cl::Error err) {
        CL_TO_AF_ERROR(err);
    }
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr)
{
    try {
        int n = opencl::getActiveDeviceId();
        auto iter = pinned_maps[n].find(ptr);

        if (iter != pinned_maps[n].end()) {
            getQueue().enqueueUnmapMemObject(pinned_maps[n][ptr], ptr);
            pinned_maps[n].erase(iter);
        }

    } catch(cl::Error err) {
        CL_TO_AF_ERROR(err);
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
    return (T *)getMemoryManager().alloc(elements * sizeof(T));
}

cl::Buffer *bufferAlloc(const size_t &bytes)
{
    return (cl::Buffer *)getMemoryManager().alloc(bytes);
}

template<typename T>
void memFree(T *ptr)
{
    return getMemoryManager().unlock((void *)ptr, false);
}

void bufferFree(cl::Buffer *buf)
{
    return getMemoryManager().unlock((void *)buf, false);
}

template<typename T>
void memFreeLocked(T *ptr, bool user_unlock)
{
    getMemoryManager().unlock((void *)ptr, user_unlock);
}

template<typename T>
void memLock(const T *ptr)
{
    getMemoryManager().userLock((void *)ptr);
}

template<typename T>
void memUnlock(const T *ptr)
{
    getMemoryManager().userUnlock((void *)ptr);
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
    return (T *)getMemoryManagerPinned().alloc(elements * sizeof(T));
}

template<typename T>
void pinnedFree(T* ptr)
{
    return getMemoryManagerPinned().unlock((void *)ptr, false);
}

#define INSTANTIATE(T)                                      \
    template T* memAlloc(const size_t &elements);           \
    template void memFree(T* ptr);                          \
    template void memFreeLocked(T* ptr, bool user_unlock);  \
    template void memLock(const T* ptr);                    \
    template void memUnlock(const T* ptr);                  \
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
