/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <err_cpu.hpp>
#include <types.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <memory>
#include <MemoryManager.hpp>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CPU_MEM_DEBUG
#define AF_CPU_MEM_DEBUG 0
#endif

namespace cpu
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
        this->garbageCollect();
    }
};

int MemoryManager::getActiveDeviceId()
{
    return cpu::getActiveDeviceId();
}

MemoryManager::MemoryManager() :
    common::MemoryManager(getDeviceCount(), MAX_BUFFERS, MAX_BYTES, AF_MEM_DEBUG || AF_CPU_MEM_DEBUG)
{}


void *MemoryManager::nativeAlloc(const size_t bytes)
{
    return malloc(bytes);
}

void MemoryManager::nativeFree(void *ptr)
{
    return free((void *)ptr);
}

static MemoryManager &getMemoryManager()
{
    static MemoryManager instance;
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

template<typename T>
void memFree(T *ptr)
{
    return getMemoryManager().unlock((void *)ptr, false);
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
    getQueue().sync();
    getMemoryManager().bufferInfo(alloc_bytes, alloc_buffers,
                                  lock_bytes,  lock_buffers);
}

template<typename T>
T* pinnedAlloc(const size_t &elements)
{
    return (T *)getMemoryManager().alloc(elements * sizeof(T));
}

template<typename T>
void pinnedFree(T* ptr)
{
    return getMemoryManager().unlock((void *)ptr, false);
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
INSTANTIATE(ushort)
INSTANTIATE(short )

}
