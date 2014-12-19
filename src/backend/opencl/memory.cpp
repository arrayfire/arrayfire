/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <platform.hpp>
#include <dispatch.hpp>
#include <map>
#include <types.hpp>

namespace opencl
{
    const int MAX_BUFFERS = 100;
    const int MAX_BYTES = (1 << 30);

    typedef struct
    {
        bool is_free;
        size_t bytes;
    } mem_info;

    static size_t used_bytes = 0;
    typedef std::map<cl::Buffer *, mem_info> mem_t;
    typedef mem_t::iterator mem_iter;
    mem_t memory_maps[DeviceManager::MAX_DEVICES];

    static void destroy(cl::Buffer *ptr)
    {
        delete ptr;
    }

    static void garbageCollect()
    {
        int n = getActiveDeviceId();
        for(mem_iter iter = memory_maps[n].begin(); iter != memory_maps[n].end(); iter++) {
            if ((iter->second).is_free) destroy(iter->first);
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

    cl::Buffer *bufferAlloc(const size_t &bytes)
    {
        int n = getActiveDeviceId();
        cl::Buffer *ptr = NULL;
        size_t alloc_bytes = divup(bytes, 1024) * 1024;

        if (bytes > 0) {

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
                    return iter->first;
                }
            }

            try {
                ptr = new cl::Buffer(getContext(), CL_MEM_READ_WRITE, alloc_bytes);
            } catch(...) {
                garbageCollect();
                ptr = new cl::Buffer(getContext(), CL_MEM_READ_WRITE, alloc_bytes);
            }

            mem_info info = {false, alloc_bytes};
            memory_maps[n][ptr] = info;
            used_bytes += alloc_bytes;
        }
        return ptr;
    }

    void bufferFree(cl::Buffer *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {
            iter->second.is_free = true;
            used_bytes -= iter->second.bytes;
        } else {
            destroy(ptr); // Free it because we are not sure what the size is
        }
    }

    template<typename T>
    T *memAlloc(const size_t &elements)
    {
        return (T *)bufferAlloc(elements * sizeof(T));
    }

    template<typename T>
    void memFree(T *ptr)
    {
        return bufferFree((cl::Buffer *)ptr);
    }


    typedef std::map<void*, cl::Buffer *> pinned_t;
    typedef pinned_t::iterator pinned_iter;
    pinned_t pinned_maps[DeviceManager::MAX_DEVICES];

    template<typename T>
    T* pinnedAlloc(const size_t &elements)
    {
        void *ptr = NULL;
        cl::Buffer *buf = new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, elements * sizeof(T));

        ptr = getQueue().enqueueMapBuffer(*buf, true, CL_MAP_READ|CL_MAP_WRITE,
                                          0, elements * sizeof(T));

        pinned_maps[getActiveDeviceId()][ptr] = buf;

        return (T*)ptr;
    }

    template<typename T>
    void pinnedFree(T* ptr)
    {
        int n = getActiveDeviceId();
        pinned_iter loc = pinned_maps[n].find((void *)ptr);
        if(loc != pinned_maps[n].end()) {
            cl::Buffer *buf = loc->second;
            getQueue().enqueueUnmapMemObject(*buf, (void *)ptr);
            delete buf;
            pinned_maps[n].erase(loc);
        }
    }

#define INSTANTIATE(T)                              \
    template T* memAlloc(const size_t &elements);   \
    template void memFree(T* ptr);                  \
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
}
