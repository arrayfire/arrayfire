/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>
#include <af/dim4.hpp>

#include <utility>

using common::bytesToString;

using af::dim4;
using std::function;
using std::move;
using std::unique_ptr;

namespace opencl {
float getMemoryPressure() { return memoryManager().getMemoryPressure(); }
float getMemoryPressureThreshold() {
    return memoryManager().getMemoryPressureThreshold();
}

bool jitTreeExceedsMemoryPressure(size_t bytes) {
    return memoryManager().jitTreeExceedsMemoryPressure(bytes);
}

void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize() { return memoryManager().getMemStepSize(); }

void signalMemoryCleanup() { memoryManager().signalMemoryCleanup(); }

void shutdownMemoryManager() { memoryManager().shutdown(); }

void shutdownPinnedMemoryManager() { pinnedMemoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc(
    const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
    auto *buf = static_cast<cl::Buffer *>(ptr);
    return unique_ptr<cl::Buffer, function<void(cl::Buffer *)>>(buf,
                                                                bufferFree);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

template<typename T>
void memFree(T *ptr) {
    return memoryManager().unlock(static_cast<void *>(ptr), false);
}

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

cl::Buffer *bufferAlloc(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), 1);
    return static_cast<cl::Buffer *>(ptr);
}

void bufferFree(cl::Buffer *buf) {
    return memoryManager().unlock(static_cast<void *>(buf), false);
}

void memLock(const void *ptr) {
    memoryManager().userLock(const_cast<void *>(ptr));
}

void memUnlock(const void *ptr) {
    memoryManager().userUnlock(const_cast<void *>(ptr));
}

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked(const_cast<void *>(ptr));
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<T *>(ptr);
}

template<typename T>
void pinnedFree(T *ptr) {
    pinnedMemoryManager().unlock(static_cast<void *>(ptr), false);
}

#define INSTANTIATE(T)                                                         \
    template unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc<T>( \
        const size_t &elements);                                               \
    template void memFree(T *ptr);                                             \
    template T *pinnedAlloc(const size_t &elements);                           \
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
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(common::half)

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        try {
            opencl::setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return opencl::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) {
    return opencl::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    auto ptr = static_cast<void *>(new cl::Buffer(
        getContext(), CL_MEM_READ_WRITE,  // NOLINT(hicpp-signed-bitwise)
        bytes));
    AF_TRACE("nativeAlloc: {} {}", bytesToString(bytes), ptr);
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    AF_TRACE("nativeFree:          {}", ptr);
    delete static_cast<cl::Buffer *>(ptr);
}

AllocatorPinned::AllocatorPinned() : pinnedMaps(opencl::getDeviceCount()) {
    logger = common::loggerFactory("mem");
}

void AllocatorPinned::shutdown() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        opencl::setDevice(n);
        shutdownPinnedMemoryManager();
        auto currIterator = pinnedMaps[n].begin();
        auto endIterator  = pinnedMaps[n].end();
        while (currIterator != endIterator) {
            pinnedMaps[n].erase(currIterator++);
        }
    }
}

int AllocatorPinned::getActiveDeviceId() { return opencl::getActiveDeviceId(); }

size_t AllocatorPinned::getMaxMemorySize(int id) {
    return opencl::getDeviceMemorySize(id);
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;
    auto *buf = new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, bytes);
    ptr = getQueue().enqueueMapBuffer(*buf, true, CL_MAP_READ | CL_MAP_WRITE, 0,
                                      bytes);
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    pinnedMaps[opencl::getActiveDeviceId()].emplace(ptr, buf);
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    int n     = opencl::getActiveDeviceId();
    auto map  = pinnedMaps[n];
    auto iter = map.find(ptr);

    if (iter != map.end()) {
        cl::Buffer *buf = map[ptr];
        getQueue().enqueueUnmapMemObject(*buf, ptr);
        delete buf;
        map.erase(iter);
    }
}
}  // namespace opencl
