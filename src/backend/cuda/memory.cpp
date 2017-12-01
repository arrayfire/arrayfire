/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <platform.hpp>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <common/util.hpp>
#include <types.hpp>
#include <common/dispatch.hpp>

#include <mutex>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CUDA_MEM_DEBUG
#define AF_CUDA_MEM_DEBUG 0
#endif

using std::lock_guard;
using std::recursive_mutex;
using std::function;
using std::unique_ptr;

namespace cuda
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
uptr<T>
memAlloc(const size_t &elements)
{
    size_t size = elements * sizeof(T);
    return uptr<T>(static_cast<T*>(memoryManager().alloc(size, false)),
                   memFree<T>);
}

void* memAllocUser(const size_t &bytes)
{
    return memoryManager().alloc(bytes, true);
}
template<typename T>
void memFree(T *ptr)
{
    memoryManager().unlock((void *)ptr, false);
}

void memFreeUser(void *ptr)
{
    memoryManager().unlock((void *)ptr, true);
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

#define INSTANTIATE(T)                                 \
    template uptr<T> memAlloc(const size_t &elements); \
    template void memFree(T* ptr);                     \
    template T* pinnedAlloc(const size_t &elements);   \
    template void pinnedFree(T* ptr);

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
    INSTANTIATE(void *)

MemoryManager::MemoryManager()
    : common::MemoryManager<cuda::MemoryManager>(getDeviceCount(), common::MAX_BUFFERS,
                                                 AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

MemoryManager::~MemoryManager()
{
    common::lock_guard_t lock(this->memory_mutex);
    for (int n = 0; n < cuda::getDeviceCount(); n++) {
        try {
            cuda::setDevice(n);
            garbageCollect();
        } catch(AfError err) {
            continue; // Do not throw any errors while shutting down
        }
    }
}

int MemoryManager::getActiveDeviceId()
{
    return cuda::getActiveDeviceId();
}

size_t MemoryManager::getMaxMemorySize(int id)
{
    return cuda::getDeviceMemorySize(id);
}

void *MemoryManager::nativeAlloc(const size_t bytes)
{
    lock_guard<recursive_mutex> lock(getDriverApiMutex(getActiveDeviceId()));
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void MemoryManager::nativeFree(void *ptr)
{
    lock_guard<recursive_mutex> lock(getDriverApiMutex(getActiveDeviceId()));
    cudaError_t err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) {
        CUDA_CHECK(err);
    }
}

MemoryManagerPinned::MemoryManagerPinned()
    : common::MemoryManager<MemoryManagerPinned>(1, common::MAX_BUFFERS,
                                                 AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

MemoryManagerPinned::~MemoryManagerPinned()
{
    garbageCollect();
}

int MemoryManagerPinned::getActiveDeviceId()
{
    return 0; // pinned uses a single vector
}

size_t MemoryManagerPinned::getMaxMemorySize(int id)
{
    return cuda::getHostMemorySize();
}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes)
{
    lock_guard<recursive_mutex> lock(getDriverApiMutex(getActiveDeviceId()));
    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr)
{
    lock_guard<recursive_mutex> lock(getDriverApiMutex(getActiveDeviceId()));
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaErrorCudartUnloading) {
        CUDA_CHECK(err);
    }
}
}
