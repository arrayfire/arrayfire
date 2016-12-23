/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_cuda.hpp>
#include <memoryManager.hpp>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

#ifndef AF_CUDA_MEM_DEBUG
#define AF_CUDA_MEM_DEBUG 0
#endif

namespace cuda
{

int MemoryManager::getActiveDeviceId()
{
    return cuda::getActiveDeviceId();
}

size_t MemoryManager::getMaxMemorySize(int id)
{
    return cuda::getDeviceMemorySize(id);
}

MemoryManager::MemoryManager() :
    common::MemoryManager(getDeviceCount(), common::MAX_BUFFERS, AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

void *MemoryManager::nativeAlloc(const size_t bytes)
{
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void MemoryManager::nativeFree(void *ptr)
{
    cudaError_t err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) {
        CUDA_CHECK(err);
    }
}

int MemoryManagerPinned::getActiveDeviceId()
{
    return 0; // pinned uses a single vector
}

size_t MemoryManagerPinned::getMaxMemorySize(int id)
{
    return cuda::getHostMemorySize();
}

MemoryManagerPinned::MemoryManagerPinned() :
    common::MemoryManager(1, common::MAX_BUFFERS, AF_MEM_DEBUG || AF_CUDA_MEM_DEBUG)
{
    this->setMaxMemorySize();
}

void *MemoryManagerPinned::nativeAlloc(const size_t bytes)
{
    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

void MemoryManagerPinned::nativeFree(void *ptr)
{
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaErrorCudartUnloading) {
        CUDA_CHECK(err);
    }
}

}
