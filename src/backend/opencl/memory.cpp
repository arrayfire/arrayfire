/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_opencl.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <types.hpp>

#include <common/Logger.hpp>
#include <common/MemoryManagerImpl.hpp>
#include <spdlog/spdlog.h>

#include <utility>

using common::bytesToString;

using std::function;
using std::move;
using std::unique_ptr;

namespace opencl {
void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize(void) { return memoryManager().getMemStepSize(); }

size_t getMaxBytes() { return memoryManager().getMaxBytes(); }

unsigned getMaxBuffers() { return memoryManager().getMaxBuffers(); }

void garbageCollect() { memoryManager().garbageCollect(); }

void shutdownMemoryManager() { memoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc(
    const size_t &elements) {
    af_buffer_info pair = memoryManager().alloc(elements * sizeof(T), false);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    if (e) e.enqueueWait(getQueue()());
    void *rawPtr;
    af_unlock_buffer_info_ptr(&rawPtr, pair);
    cl::Buffer *ptr = static_cast<cl::Buffer *>(rawPtr);
    af_delete_buffer_info(pair);
    return unique_ptr<cl::Buffer, function<void(cl::Buffer *)>>(ptr,
                                                                bufferFree);
}

void *memAllocUser(const size_t &bytes) {
    af_buffer_info pair = memoryManager().alloc(bytes, true);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    if (e) e.enqueueWait(getQueue()());
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return ptr;
}

template<typename T>
void memFree(T *ptr) {
    return memoryManager().unlock((void *)ptr, detail::createAndMarkEvent(),
                                  false);
}

void memFreeUser(void *ptr) {
    memoryManager().unlock((void *)ptr, detail::createAndMarkEvent(), true);
}

cl::Buffer *bufferAlloc(const size_t &bytes) {
    af_buffer_info pair = memoryManager().alloc(bytes, false);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    if (e) e.enqueueWait(getQueue()());
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return static_cast<cl::Buffer *>(ptr);
}

void bufferFree(cl::Buffer *buf) {
    return memoryManager().unlock((void *)buf, detail::createAndMarkEvent(),
                                  false);
}

void memLock(const void *ptr) { memoryManager().userLock((void *)ptr); }

void memUnlock(const void *ptr) { memoryManager().userUnlock((void *)ptr); }

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked((void *)ptr);
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    af_buffer_info pair = memoryManager().alloc(elements * sizeof(T), false);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    if (e) e.enqueueWait(getQueue()());
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return static_cast<T *>(ptr);
}

template<typename T>
void pinnedFree(T *ptr) {
    pinnedMemoryManager().unlock((void *)ptr, detail::createAndMarkEvent(),
                                 false);
}

bool checkMemoryLimit() { return memoryManager().checkMemoryLimit(); }

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

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

Allocator::~Allocator() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        try {
            opencl::setDevice(n);
            shutdownMemoryManager();
        } catch (AfError err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return opencl::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) {
    return opencl::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    auto ptr = (void *)(new cl::Buffer(getContext(), CL_MEM_READ_WRITE, bytes));
    AF_TRACE("nativeAlloc: {} {}", bytesToString(bytes), ptr);
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    AF_TRACE("nativeFree:          {}", ptr);
    delete (cl::Buffer *)ptr;
}

AllocatorPinned::AllocatorPinned() { logger = common::loggerFactory("mem"); }

AllocatorPinned::~AllocatorPinned() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        opencl::setDevice(n);
        shutdownMemoryManager();
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
    cl::Buffer *buf =
        new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, bytes);
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
