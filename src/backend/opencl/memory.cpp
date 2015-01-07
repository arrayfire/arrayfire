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
    static void garbageCollect();
    static void pinnedGarbageCollect();

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
            for(int i = 0; i < (int)getDeviceCount(); i++) {
                setDevice(i);
                garbageCollect();
                pinnedGarbageCollect();
            }
        }
    };

    bool Manager::initialized = false;

    static void managerInit()
    {
        if(Manager::initialized == false)
            static Manager pm = Manager();
    }

    static const unsigned MAX_BUFFERS   = 100;
    static const unsigned MAX_BYTES     = (1 << 30);

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
        managerInit();
        return (T *)bufferAlloc(elements * sizeof(T));
    }

    template<typename T>
    void memFree(T *ptr)
    {
        return bufferFree((cl::Buffer *)ptr);
    }

    // pinned memory manager
    typedef struct {
        cl::Buffer *buf;
        mem_info info;
    } pinned_info;

    typedef std::map<void*, pinned_info> pinned_t;
    typedef pinned_t::iterator pinned_iter;
    pinned_t pinned_maps[DeviceManager::MAX_DEVICES];
    static size_t pinned_used_bytes = 0;

    static void pinnedDestroy(void *ptr)
    {
        int n = getActiveDeviceId();
        pinned_iter loc = pinned_maps[n].find(ptr);

        if(loc != pinned_maps[n].end()) {
            cl::Buffer *buf = loc->second.buf;
            getQueue().enqueueUnmapMemObject(*buf, (void *)ptr);
            delete buf;
            pinned_maps[n].erase(loc);
        }
    }

    static void pinnedGarbageCollect()
    {
        int n = getActiveDeviceId();
        for(pinned_iter iter = pinned_maps[n].begin(); iter != pinned_maps[n].end(); iter++) {
            if ((iter->second).info.is_free) pinnedDestroy(iter->first);
        }

        pinned_iter memory_curr = pinned_maps[n].begin();
        pinned_iter memory_end  = pinned_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.info.is_free) {
                pinned_maps[n].erase(memory_curr++);
            } else {
                ++memory_curr;
            }
        }
    }

    void *pinnedBufferAlloc(const size_t &bytes)
    {
        void *ptr = NULL;
        int n = getActiveDeviceId();
        cl::Buffer *buf = NULL;
        // Allocate the higher megabyte. Overhead of creating pinned memory is
        // more so we want more resuable memory.
        size_t alloc_bytes = divup(bytes, 1048576) * 1048576;

        if (bytes > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (pinned_maps[n].size() >= MAX_BUFFERS || pinned_used_bytes >= MAX_BYTES) {
                pinnedGarbageCollect();
            }

            for(pinned_iter iter = pinned_maps[n].begin();
                iter != pinned_maps[n].end(); iter++) {

                mem_info info = iter->second.info;
                if (info.is_free && info.bytes == alloc_bytes) {
                    iter->second.info.is_free = false;
                    pinned_used_bytes += alloc_bytes;
                    return iter->first;
                }
            }

            try {
                buf = new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, alloc_bytes);

                ptr = getQueue().enqueueMapBuffer(*buf, true, CL_MAP_READ|CL_MAP_WRITE,
                                                  0, alloc_bytes);
            } catch(...) {
                pinnedGarbageCollect();
                buf = new cl::Buffer(getContext(), CL_MEM_ALLOC_HOST_PTR, alloc_bytes);

                ptr = getQueue().enqueueMapBuffer(*buf, true, CL_MAP_READ|CL_MAP_WRITE,
                                                  0, alloc_bytes);
            }
            mem_info info = {false, alloc_bytes};
            pinned_info pt = {buf, info};
            pinned_maps[n][ptr] = pt;
            pinned_used_bytes += alloc_bytes;
        }
        return ptr;
    }

    void pinnedBufferFree(void *ptr)
    {
        int n = getActiveDeviceId();
        pinned_iter iter = pinned_maps[n].find(ptr);

        if (iter != pinned_maps[n].end()) {
            iter->second.info.is_free = true;
            pinned_used_bytes -= iter->second.info.bytes;
        } else {
            pinnedDestroy(ptr); // Free it because we are not sure what the size is
        }
    }

    template<typename T>
    T* pinnedAlloc(const size_t &elements)
    {
        managerInit();
        return (T *)pinnedBufferAlloc(elements * sizeof(T));
    }

    template<typename T>
    void pinnedFree(T* ptr)
    {
        return pinnedBufferFree((void *) ptr);
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
