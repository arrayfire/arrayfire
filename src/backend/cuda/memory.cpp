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
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/util.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <err_cuda.hpp>
#include <memory_manager_impl.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>

#include <mutex>
#include <utility>

using common::bytesToString;
using common::half;

using std::move;

namespace cuda {
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
uptr<T> memAlloc(const size_t &elements) {
    size_t size         = elements * sizeof(T);
    af_buffer_info pair = memoryManager().alloc(elements * sizeof(T), false);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    cudaStream_t stream = getActiveStream();
    if (e) e.enqueueWait(stream);
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return uptr<T>(static_cast<T *>(ptr), memFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    af_buffer_info pair = memoryManager().alloc(bytes, true);
    detail::Event e     = std::move(getEventFromBufferInfoHandle(pair));
    cudaStream_t stream = getActiveStream();
    if (e) e.enqueueWait(stream);
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return ptr;
}

template<typename T>
void memFree(T *ptr) {
    memoryManager().unlock((void *)ptr, detail::createAndMarkEvent(), false);
}

void memFreeUser(void *ptr) {
    memoryManager().unlock((void *)ptr, detail::createAndMarkEvent(), true);
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
    cudaStream_t stream = getActiveStream();
    if (e) e.enqueueWait(stream);
    void *ptr;
    af_unlock_buffer_info_ptr(&ptr, pair);
    af_delete_buffer_info(pair);
    return (T *)ptr;
}

template<typename T>
void pinnedFree(T *ptr) {
    pinnedMemoryManager().unlock((void *)ptr, detail::createAndMarkEvent(),
                                 false);
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
INSTANTIATE(half)

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

Allocator::~Allocator() {
    for (int n = 0; n < cuda::getDeviceCount(); n++) {
        try {
            cuda::setDevice(n);
            // TODO: figure out something reasonable to do here. The destruction
            // order is very weird and bad and needs to be fixed
            shutdownMemoryManager();
        } catch (AfError err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return cuda::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) {
    return cuda::getDeviceMemorySize(id);
}

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

AllocatorPinned::~AllocatorPinned() { garbageCollect(); }

int AllocatorPinned::getActiveDeviceId() {
    return 0;  // pinned uses a single vector
}

size_t AllocatorPinned::getMaxMemorySize(int id) {
    UNUSED(id);
    return cuda::getHostMemorySize();
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
