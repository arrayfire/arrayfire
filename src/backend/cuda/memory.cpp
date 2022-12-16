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
#include <common/MemoryManagerBase.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/util.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>
#include <af/dim4.hpp>

#include <cstdlib>
#include <mutex>

using af::dim4;
using arrayfire::common::bytesToString;
using arrayfire::common::half;

using std::move;

namespace arrayfire {
namespace cuda {
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
uptr<T> memAlloc(const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return uptr<T>(static_cast<T *>(ptr), memFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

template<typename T>
void memFree(T *ptr) {
    memoryManager().unlock(static_cast<void *>(ptr), false);
}

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

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
INSTANTIATE(half)

template void memFree(void *ptr);

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < getDeviceCount(); n++) {
        try {
            setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return cuda::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) { return getDeviceMemorySize(id); }

void *Allocator::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    AF_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    AF_TRACE("nativeFree:          {}", ptr);
    cudaError_t err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}

AllocatorPinned::AllocatorPinned() { logger = common::loggerFactory("mem"); }

void AllocatorPinned::shutdown() { shutdownPinnedMemoryManager(); }

int AllocatorPinned::getActiveDeviceId() {
    return 0;  // pinned uses a single vector
}

size_t AllocatorPinned::getMaxMemorySize(int id) {
    UNUSED(id);
    return getHostMemorySize();
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}
}  // namespace cuda
}  // namespace arrayfire
