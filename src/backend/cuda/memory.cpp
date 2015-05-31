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
    static size_t memory_resolution = 1024; //1KB

    void setMemStepSize(size_t step_bytes)
    {
        memory_resolution = step_bytes;
    }

    size_t getMemStepSize(void)
    {
        return memory_resolution;
    }

    template<typename T>
    static void cudaFreeWrapper(T *ptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaErrorCudartUnloading) // see issue #167
            CUDA_CHECK(err);
    }

    template<typename T>
    static void pinnedFreeWrapper(T *ptr)
    {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaErrorCudartUnloading) // see issue #167
            CUDA_CHECK(err);
    }

#ifdef AF_CUDA_MEM_DEBUG

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

    template<typename T>
    T* pinnedAlloc(const size_t &elements)
    {
        T* ptr = NULL;
        CUDA_CHECK(cudaMallocHost((void **)(&ptr), elements * sizeof(T)));
        return (T*)ptr;
    }

    template<typename T>
    void pinnedFree(T *ptr)
    {
        pinnedFreeWrapper(ptr); // Free it because we are not sure what the size is
    }

#else

    // Manager Class
    // Dummy used to call garbage collection at the end of the program
    class Manager
    {
        public:
        static bool initialized;
        Manager()
        {
            initialized = true;
        }

        ~Manager()
        {
            // Destructors should not through exceptions
            try {
                for(int i = 0; i < getDeviceCount(); i++) {
                    setDevice(i);
                    garbageCollect();
                }
                pinnedGarbageCollect();

            } catch (AfError &ex) {

                const char* perr = getenv("AF_PRINT_ERRORS");
                if(perr && perr[0] != '0') {
                    fprintf(stderr, "%s\n", ex.what());
                }
            }
        }
    };

    bool Manager::initialized = false;

    static void managerInit()
    {
        if(Manager::initialized == false)
            static Manager pm = Manager();
    }

    typedef struct
    {
        bool is_free;
        bool is_unlinked;
        size_t bytes;
    } mem_info;

    static size_t used_bytes[DeviceManager::MAX_DEVICES] = {0};
    static size_t used_buffers[DeviceManager::MAX_DEVICES] = {0};
    static size_t total_bytes[DeviceManager::MAX_DEVICES] = {0};
    typedef std::map<void *, mem_info> mem_t;
    typedef mem_t::iterator mem_iter;

    mem_t memory_maps[DeviceManager::MAX_DEVICES];

    void garbageCollect()
    {
        int n = getActiveDeviceId();

        for(mem_iter iter = memory_maps[n].begin();
            iter != memory_maps[n].end(); ++iter) {

            if ((iter->second).is_free) {

                if (!(iter->second).is_unlinked) {
                    cudaFreeWrapper(iter->first);
                }
                total_bytes[n] -= iter->second.bytes;
            }
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
        managerInit();
        int n = getActiveDeviceId();
        T* ptr = NULL;
        size_t alloc_bytes = divup(sizeof(T) * elements, memory_resolution) * memory_resolution;

        if (elements > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (memory_maps[n].size() >= MAX_BUFFERS || used_bytes[n] >= MAX_BYTES) {
                garbageCollect();
            }

            for(mem_iter iter = memory_maps[n].begin();
                iter != memory_maps[n].end(); ++iter) {

                mem_info info = iter->second;

                if (  info.is_free &&
                     !info.is_unlinked &&
                      info.bytes == alloc_bytes) {

                    iter->second.is_free = false;
                    used_bytes[n] += alloc_bytes;
                    used_buffers[n]++;
                    return (T *)iter->first;
                }
            }

            // Perform garbage collection if memory can not be allocated
            if (cudaMalloc((void **)&ptr, alloc_bytes) != cudaSuccess) {
                garbageCollect();
                CUDA_CHECK(cudaMalloc((void **)(&ptr), alloc_bytes));
            }

            mem_info info = {false, false, alloc_bytes};
            memory_maps[n][ptr] = info;
            used_bytes[n] += alloc_bytes;
            used_buffers[n]++;
            total_bytes[n] += alloc_bytes;
        }
        return ptr;
    }

    template<typename T>
    void memFree(T *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find((void *)ptr);

        if (iter != memory_maps[n].end()) {

            if ((iter->second).is_unlinked) return;

            iter->second.is_free = true;
            used_bytes[n] -= iter->second.bytes;
            used_buffers[n]--;

        } else {
            cudaFreeWrapper(ptr); // Free it because we are not sure what the size is
        }
    }

    template<typename T>
    void memUnlink(T *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find((void *)ptr);

        if (iter != memory_maps[n].end()) {

            iter->second.is_free = true;
            iter->second.is_unlinked = true;
            used_bytes[n] -= iter->second.bytes;
            used_buffers[n]--;

        } else {
            mem_info info = { false,
                              false,
                              100 }; //This number is not relevant

            memory_maps[n][ptr] = info;
        }
    }

    void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes,  size_t *lock_buffers)
    {
        int n = getActiveDeviceId();
        if (alloc_bytes   ) *alloc_bytes   = total_bytes[n];
        if (alloc_buffers ) *alloc_buffers = memory_maps[n].size();
        if (lock_bytes    ) *lock_bytes    = used_bytes[n];
        if (lock_buffers  ) *lock_buffers  = used_buffers[n];
    }

    //////////////////////////////////////////////////////////////////////////////
    mem_t pinned_maps;
    static size_t pinned_used_bytes = 0;

    void pinnedGarbageCollect()
    {
        for(mem_iter iter = pinned_maps.begin(); iter != pinned_maps.end(); ++iter) {
            if ((iter->second).is_free) {
                pinnedFreeWrapper(iter->first);
            }
        }

        mem_iter memory_curr = pinned_maps.begin();
        mem_iter memory_end  = pinned_maps.end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.is_free) {
                pinned_maps.erase(memory_curr++);
            } else {
                ++memory_curr;
            }
        }
    }

    template<typename T>
    T* pinnedAlloc(const size_t &elements)
    {
        managerInit();
        T* ptr = NULL;
        // Allocate the higher megabyte. Overhead of creating pinned memory is
        // more so we want more resuable memory.
        size_t alloc_bytes = divup(sizeof(T) * elements, 1048576) * 1048576;

        if (elements > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (pinned_maps.size() >= MAX_BUFFERS || pinned_used_bytes >= MAX_BYTES) {
                pinnedGarbageCollect();
            }

            for(mem_iter iter = pinned_maps.begin();
                iter != pinned_maps.end(); ++iter) {

                mem_info info = iter->second;
                if (info.is_free && info.bytes == alloc_bytes) {
                    iter->second.is_free = false;
                    pinned_used_bytes += alloc_bytes;
                    return (T *)iter->first;
                }
            }

            // Perform garbage collection if memory can not be allocated
            if (cudaMallocHost((void **)&ptr, alloc_bytes) != cudaSuccess) {
                pinnedGarbageCollect();
                CUDA_CHECK(cudaMallocHost((void **)(&ptr), alloc_bytes));
            }

            mem_info info = {false, false, alloc_bytes};
            pinned_maps[ptr] = info;
            pinned_used_bytes += alloc_bytes;
        }
        return (T*)ptr;
    }

    template<typename T>
    void pinnedFree(T *ptr)
    {
        mem_iter iter = pinned_maps.find((void *)ptr);

        if (iter != pinned_maps.end()) {
            iter->second.is_free = true;
            pinned_used_bytes -= iter->second.bytes;
        } else {
            pinnedFreeWrapper(ptr); // Free it because we are not sure what the size is
        }
    }

#endif

#define INSTANTIATE(T)                              \
    template T* memAlloc(const size_t &elements);   \
    template void memFree(T* ptr);                  \
    template void memUnlink(T* ptr);                \
    template T* pinnedAlloc(const size_t &elements);\
    template void pinnedFree(T* ptr);               \

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

}
