/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <platform.hpp>
#include <types.hpp>
#include <err_opencl.hpp>

#include <common/MemoryManagerImpl.hpp>

template class common::MemoryManager<opencl::MemoryManager>;
template class common::MemoryManager<opencl::MemoryManagerPinned>;

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_OPENCL_MEM_DEBUG
#define AF_OPENCL_MEM_DEBUG 0
#endif

using std::unique_ptr;
using std::function;

namespace opencl
{
void setMemStepSize(size_t step_bytes)
{
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize(void)
{
    return memoryManager().getMemStepSize();
}

size_t getMaxBytes()
{
    return memoryManager().getMaxBytes();
}

unsigned getMaxBuffers()
{
    return memoryManager().getMaxBuffers();
}

void garbageCollect()
{
    memoryManager().garbageCollect();
}

void printMemInfo(const char *msg, const int device)
{
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<T[], function<void(T *)>>
memAlloc(const size_t &elements)
{
    T* ptr = (T *)memoryManager().alloc(elements * sizeof(T), false);
    return unique_ptr<T[], function<void(T *)>>(ptr, memFree<T>);
}

void* memAllocUser(const size_t &bytes)
{
    return memoryManager().alloc(bytes, true);
}
template<typename T>
void memFree(T *ptr)
{
    return memoryManager().unlock((void *)ptr, false);
}

void memFreeUser(void *ptr)
{
    memoryManager().unlock((void *)ptr, true);
}

cl::Buffer *bufferAlloc(const size_t &bytes)
{
    return (cl::Buffer *)memoryManager().alloc(bytes, false);
}

void bufferFree(cl::Buffer *buf)
{
    return memoryManager().unlock((void *)buf, false);
}

void memLock(const void *ptr)
{
    memoryManager().userLock((void *)ptr);
}

void memUnlock(const void *ptr)
{
    memoryManager().userUnlock((void *)ptr);
}

bool isLocked(const void *ptr)
{
    return memoryManager().isUserLocked((void *)ptr);
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes,  size_t *lock_buffers)
{
    memoryManager().bufferInfo(alloc_bytes, alloc_buffers,
                                  lock_bytes,  lock_buffers);
}

template<typename T>
T* pinnedAlloc(const size_t &elements)
{
    return (T *)pinnedMemoryManager().alloc(elements * sizeof(T), false);
}

template<typename T>
void pinnedFree(T* ptr)
{
    return pinnedMemoryManager().unlock((void *)ptr, false);
}

bool checkMemoryLimit()
{
    return memoryManager().checkMemoryLimit();
}

#define INSTANTIATE(T)                                                              \
    template unique_ptr<T[], function<void(T *)>> memAlloc(const size_t &elements); \
    template void memFree(T* ptr);                                                  \
    template T* pinnedAlloc(const size_t &elements);                                \
    template void pinnedFree(T* ptr);                                               \

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

MemoryManager::MemoryManager()
    : common::MemoryManager<opencl::MemoryManager>(getDeviceCount(), common::MAX_BUFFERS,
                                                   AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

MemoryManager::~MemoryManager()
{
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        try {
            opencl::setDevice(n);
            this->garbageCollect();
        } catch(AfError err) {
            continue; // Do not throw any errors while shutting down
        }
    }
}

int MemoryManager::getActiveDeviceId()
{
    return opencl::getActiveDeviceId();
}

size_t MemoryManager::getMaxMemorySize(int id)
{
    return opencl::getDeviceMemorySize(id);
}

void *MemoryManager::nativeAlloc(const size_t bytes)
{
    return (void *)(new cl::Buffer(getContext(), CL_MEM_READ_WRITE, bytes));
}

void MemoryManager::nativeFree(void *ptr)
{
    delete (cl::Buffer *)ptr;
}

MemoryManagerPinned::MemoryManagerPinned()
    : common::MemoryManager<MemoryManagerPinned>(getDeviceCount(), common::MAX_BUFFERS,
                                                 AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG),
    pinnedMaps(getDeviceCount())
{
    this->setMaxMemorySize();
}

MemoryManagerPinned::~MemoryManagerPinned()
{
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        opencl::setDevice(n);
        this->garbageCollect();
        auto currIterator = pinnedMaps[n].begin();
        auto endIterator  = pinnedMaps[n].end();
        while (currIterator != endIterator) {
            pinnedMaps[n].erase(currIterator++);
        }
    }
}

int MemoryManagerPinned::getActiveDeviceId()
{
    return opencl::getActiveDeviceId();
}

size_t MemoryManagerPinned::getMaxMemorySize(int id)
{
    return opencl::getDeviceMemorySize(id);
}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes)
{
    void *ptr = NULL;
    cl::Buffer* buf = new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, bytes);
    ptr = getQueue().enqueueMapBuffer(*buf, true, CL_MAP_READ | CL_MAP_WRITE, 0, bytes);
    pinnedMaps[opencl::getActiveDeviceId()].emplace(ptr, buf);
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr)
{
    int n = opencl::getActiveDeviceId();
    auto map = pinnedMaps[n];
    auto iter = map.find(ptr);

    if (iter != map.end()) {
        cl::Buffer* buf = map[ptr];
        getQueue().enqueueUnmapMemObject(*buf, ptr);
        delete buf;
        map.erase(iter);
    }
}
}
