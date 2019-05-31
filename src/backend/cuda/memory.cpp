/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>

#include <Event.hpp>
#include <common/Logger.hpp>
#include <common/MemoryManagerImpl.hpp>
#include <common/dispatch.hpp>
#include <common/util.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>

#include <mutex>

template class common::MemoryManager<cuda::MemoryManager>;
template class common::MemoryManager<cuda::MemoryManagerPinned>;

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CUDA_MEM_DEBUG
#define AF_CUDA_MEM_DEBUG 0
#endif

using common::bytesToString;
using common::MemoryEventPair;

using std::move;

namespace cuda {
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
uptr<T> memAlloc(const size_t &elements) {
    size_t size                = elements * sizeof(T);
    MemoryEventPair me = memoryManager().alloc(size, false);
    cudaStream_t stream        = getActiveStream();
    if (me.e) me.e.enqueueWait(stream);
    return uptr<T>(static_cast<T *>(me.ptr), memFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    MemoryEventPair me = memoryManager().alloc(bytes, true);
    cudaStream_t stream        = getActiveStream();
    if (me.e) me.e.enqueueWait(stream);
    return me.ptr;
}

template<typename T>
void memFree(T *ptr) {
    Event e = make_event(getActiveStream());
    memoryManager().unlock((void *)ptr, move(e), false);
}

void memFreeUser(void *ptr) {
    Event e = make_event(getActiveStream());
    memoryManager().unlock((void *)ptr, move(e), true);
}

void memLock(const void *ptr) { memoryManager().userLock((void *)ptr); }

void memUnlock(const void *ptr) { memoryManager().userUnlock((void *)ptr); }

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked((void *)ptr);
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().bufferInfo(alloc_bytes, alloc_buffers, lock_bytes,
                               lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    MemoryEventPair me =
        pinnedMemoryManager().alloc(elements * sizeof(T), false);
    cudaStream_t stream = getActiveStream();
    if (me.e) me.e.enqueueWait(stream);
    return (T *)me.ptr;
}

template<typename T>
void pinnedFree(T *ptr) {
    Event e = make_event(getActiveStream());
    return pinnedMemoryManager().unlock((void *)ptr, move(e), false);
}

bool checkMemoryLimit() { return memoryManager().checkMemoryLimit(); }

#define INSTANTIATE(T)                                 \
    template uptr<T> memAlloc(const size_t &elements); \
    template void memFree(T *ptr);                     \
    template T *pinnedAlloc(const size_t &elements);   \
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

MemoryManager::MemoryManager()
    : common::MemoryManager<cuda::MemoryManager>(
          getDeviceCount(), common::MAX_BUFFERS,
          AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG) {
    this->setMaxMemorySize();
}

MemoryManager::~MemoryManager() {
    for (int n = 0; n < cuda::getDeviceCount(); n++) {
        try {
            cuda::setDevice(n);
            garbageCollect();
        } catch (AfError err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int MemoryManager::getActiveDeviceId() { return cuda::getActiveDeviceId(); }

size_t MemoryManager::getMaxMemorySize(int id) {
    return cuda::getDeviceMemorySize(id);
}

void *MemoryManager::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    AF_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void MemoryManager::nativeFree(void *ptr) {
    AF_TRACE("nativeFree:          {}", ptr);
    cudaError_t err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}

MemoryManagerPinned::MemoryManagerPinned()
    : common::MemoryManager<MemoryManagerPinned>(
          1, common::MAX_BUFFERS, AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG) {
    this->setMaxMemorySize();
}

MemoryManagerPinned::~MemoryManagerPinned() { garbageCollect(); }

int MemoryManagerPinned::getActiveDeviceId() {
    return 0;  // pinned uses a single vector
}

size_t MemoryManagerPinned::getMaxMemorySize(int id) {
    UNUSED(id);
    return cuda::getHostMemorySize();
}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes) {
    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}
}  // namespace cuda
