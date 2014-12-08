/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <err_cuda.hpp>
#include <types.hpp>
#include <map>
#include <dispatch.hpp>
#include <platform.hpp>

namespace cuda
{
    template<typename T>
    static void cudaFreeWrapper(T *ptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaErrorCudartUnloading) // see issue #167
            CUDA_CHECK(err);
    }

#ifdef CUDA_MEM_DEBUG

    template<typename T>
    T* memAlloc(const size_t &elements)
    {
        T* ptr = NULL;
        CUDA_CHECK(cudaMalloc(&ptr, elements * sizeof(T)));
        return ptr;
    }

    template<typename T>
    void memFree(T *ptr)
    {
        cudaFreeWrapper(ptr); // Free it because we are not sure what the size is
    }

#else

    const int MAX_BUFFERS = 100;
    const int MAX_BYTES = (1 << 30);

    typedef struct
    {
        bool is_free;
        size_t bytes;
    } mem_info;

    static size_t used_bytes = 0;
    typedef std::map<void *, mem_info> mem_t;
    typedef mem_t::iterator mem_iter;

    mem_t memory_maps[DeviceManager::MAX_DEVICES];

    static void garbageCollect()
    {
        int n = getActiveDeviceId();
        for(mem_iter iter = memory_maps[n].begin(); iter != memory_maps[n].end(); iter++) {
            if ((iter->second).is_free) cudaFreeWrapper(iter->first);
        }

        mem_iter memory_curr = memory_maps[n].begin();
        mem_iter memory_end  = memory_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.is_free) {
                memory_maps[n].erase(memory_curr++);
            } else {
                ++memory_curr;
            }
        }
    }

    template<typename T>
    T* memAlloc(const size_t &elements)
    {
        int n = getActiveDeviceId();
        T* ptr = NULL;
        size_t alloc_bytes = divup(sizeof(T) * elements, 1024) * 1024;

        if (elements > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (memory_maps[n].size() >= MAX_BUFFERS || used_bytes >= MAX_BYTES) {
                garbageCollect();
            }

            for(mem_iter iter = memory_maps[n].begin();
                iter != memory_maps[n].end(); iter++) {

                mem_info info = iter->second;
                if (info.is_free && info.bytes == alloc_bytes) {
                    iter->second.is_free = false;
                    used_bytes += alloc_bytes;
                    return (T *)iter->first;
                }
            }

            // Perform garbage collection if memory can not be allocated
            if (cudaMalloc((void **)&ptr, alloc_bytes) != cudaSuccess) {
                garbageCollect();
                CUDA_CHECK(cudaMalloc((void **)(&ptr), alloc_bytes));
            }

            mem_info info = {false, alloc_bytes};
            memory_maps[n][ptr] = info;
            used_bytes += alloc_bytes;
        }
        return ptr;
    }

    template<typename T>
    void memFree(T *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find((void *)ptr);

        if (iter != memory_maps[n].end()) {
            iter->second.is_free = true;
            used_bytes -= iter->second.bytes;
        } else {
            cudaFreeWrapper(ptr); // Free it because we are not sure what the size is
        }
    }
#endif

#define INSTANTIATE(T)                              \
    template T* memAlloc(const size_t &elements); \
    template void memFree(T* ptr);                \

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
