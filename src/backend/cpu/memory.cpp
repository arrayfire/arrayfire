/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>

#include <common/Logger.hpp>
#include <common/MemoryManagerImpl.hpp>
#include <common/half.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>

#include <utility>

template class common::MemoryManager<cpu::MemoryManager>;

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CPU_MEM_DEBUG
#define AF_CPU_MEM_DEBUG 0
#endif

using common::bytesToString;
using common::half;
using std::function;
using std::move;
using std::unique_ptr;

namespace cpu {
void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize(void) { return memoryManager().getMemStepSize(); }

size_t getMaxBytes() { return memoryManager().getMaxBytes(); }

unsigned getMaxBuffers() { return memoryManager().getMaxBuffers(); }

void garbageCollect() { memoryManager().garbageCollect(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<T[], function<void(T *)>> memAlloc(const size_t &elements) {
    T *ptr = nullptr;

    common::MemoryEventPair me = memoryManager().alloc(elements * sizeof(T), false);
    if(me.e) me.e.enqueueWait(getQueue());
    ptr = (T *)me.ptr;
    return unique_ptr<T[], function<void(T *)>>(ptr, memFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    void *ptr = nullptr;
    common::MemoryEventPair me = memoryManager().alloc(bytes, true);
    if (me.e) me.e.enqueueWait(getQueue());
    return me.ptr;
}

template<typename T>
void memFree(T *ptr) {
    Event e = make_event(getQueue());
    return memoryManager().unlock((void *)ptr, move(e), false);
}

void memFreeUser(void *ptr) {
    Event e = make_event(getQueue());
    memoryManager().unlock((void *)ptr, move(e), true);
}

void memLock(const void *ptr) { memoryManager().userLock((void *)ptr); }

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked((void *)ptr);
}

void memUnlock(const void *ptr) { memoryManager().userUnlock((void *)ptr); }

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().bufferInfo(alloc_bytes, alloc_buffers, lock_bytes,
                               lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    common::MemoryEventPair me = memoryManager().alloc(elements * sizeof(T), false);
    if (me.e) me.e.enqueueWait(getQueue());
    return (T*)me.ptr;
}

template<typename T>
void pinnedFree(T *ptr) {
    Event e = make_event(getQueue());
    return memoryManager().unlock((void *)ptr, move(e), false);
}

bool checkMemoryLimit() { return memoryManager().checkMemoryLimit(); }

#define INSTANTIATE(T)                                                \
    template std::unique_ptr<T[], std::function<void(T *)>> memAlloc( \
        const size_t &elements);                                      \
    template void memFree(T *ptr);                                    \
    template T *pinnedAlloc(const size_t &elements);                  \
    template void pinnedFree(T *ptr);

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
INSTANTIATE(short)
INSTANTIATE(half)

MemoryManager::MemoryManager()
    : common::MemoryManager<cpu::MemoryManager>(
          getDeviceCount(), common::MAX_BUFFERS,
          AF_MEM_DEBUG || AF_CPU_MEM_DEBUG) {
    this->setMaxMemorySize();
}

MemoryManager::~MemoryManager() {
    for (int n = 0; n < cpu::getDeviceCount(); n++) {
        try {
            cpu::setDevice(n);
            garbageCollect();
        } catch (AfError err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int MemoryManager::getActiveDeviceId() { return cpu::getActiveDeviceId(); }

size_t MemoryManager::getMaxMemorySize(int id) {
    return cpu::getDeviceMemorySize(id);
}

void *MemoryManager::nativeAlloc(const size_t bytes) {
    void *ptr = malloc(bytes);
    AF_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    if (!ptr) AF_ERROR("Unable to allocate memory", AF_ERR_NO_MEM);
    return ptr;
}

void MemoryManager::nativeFree(void *ptr) {
    AF_TRACE("nativeFree: {: >8} {}", " ", ptr);
    // Make sure this pointer is not being used on the queue before freeing the
    // memory.
    getQueue().sync();
    return free((void *)ptr);
}
}  // namespace cpu
