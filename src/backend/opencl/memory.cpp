/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>

#include <memoryManager.hpp>
#include <platform.hpp>
#include <types.hpp>

namespace opencl
{

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
