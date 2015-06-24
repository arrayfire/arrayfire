/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <dispatch.hpp>
#include <map>
#include <types.hpp>

namespace opencl
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

    typedef struct
    {
        bool is_free;
        bool is_unlinked;
        size_t bytes;
    } mem_info;

    static size_t used_bytes[DeviceManager::MAX_DEVICES] = {0};
    static size_t used_buffers[DeviceManager::MAX_DEVICES] = {0};
    static size_t total_bytes[DeviceManager::MAX_DEVICES] = {0};

    typedef std::map<cl::Buffer *, mem_info> mem_t;
    typedef mem_t::iterator mem_iter;
    mem_t memory_maps[DeviceManager::MAX_DEVICES];

    static void destroy(cl::Buffer *ptr)
    {
        delete ptr;
    }

    void garbageCollect()
    {
        int n = getActiveDeviceId();
        for(mem_iter iter = memory_maps[n].begin();
            iter != memory_maps[n].end(); ++iter) {

            if ((iter->second).is_free) {

                if (!(iter->second).is_unlinked) {
                    destroy(iter->first);
                    total_bytes[n] -= iter->second.bytes;
                }
            }
        }

        mem_iter memory_curr = memory_maps[n].begin();
        mem_iter memory_end  = memory_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.is_free  && !memory_curr->second.is_unlinked) {
                memory_curr = memory_maps[n].erase(memory_curr);
            } else {
                ++memory_curr;
            }
        }
    }

    cl::Buffer *bufferAlloc(const size_t &bytes)
    {
        int n = getActiveDeviceId();
        cl::Buffer *ptr = NULL;
        size_t alloc_bytes = divup(bytes, memory_resolution) * memory_resolution;

        if (bytes > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (memory_maps[n].size() >= MAX_BUFFERS || used_bytes[n] >= MAX_BYTES) {
                garbageCollect();
            }

            for(mem_iter iter = memory_maps[n].begin();
                iter != memory_maps[n].end(); ++iter) {

                mem_info info = iter->second;

                if ( info.is_free &&
                    !info.is_unlinked &&
                     info.bytes == alloc_bytes) {

                    iter->second.is_free = false;
                    used_bytes[n] += alloc_bytes;
                    used_buffers[n]++;
                    return iter->first;
                }
            }

            try {
                ptr = new cl::Buffer(getContext(), CL_MEM_READ_WRITE, alloc_bytes);
            } catch(...) {
                garbageCollect();
                ptr = new cl::Buffer(getContext(), CL_MEM_READ_WRITE, alloc_bytes);
            }

            mem_info info = {false, false, alloc_bytes};
            memory_maps[n][ptr] = info;
            used_bytes[n] += alloc_bytes;
            used_buffers[n]++;
            total_bytes[n] += alloc_bytes;
        }
        return ptr;
    }

    void bufferFree(cl::Buffer *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {

            iter->second.is_free = true;
            if ((iter->second).is_unlinked) return;

            used_bytes[n] -= iter->second.bytes;
            used_buffers[n]--;
        } else {
            destroy(ptr); // Free it because we are not sure what the size is
        }
    }

    void bufferPop(cl::Buffer *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {
            iter->second.is_unlinked = true;
        } else {

            mem_info info = { false,
                              false,
                              100 }; //This number is not relevant

            memory_maps[n][ptr] = info;
        }
    }

    void bufferPush(cl::Buffer *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {
            iter->second.is_unlinked = false;
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

    template<typename T>
    void memPop(const T *ptr)
    {
        return bufferPop((cl::Buffer *)ptr);
    }

    template<typename T>
    void memPush(const T *ptr)
    {
        return bufferPush((cl::Buffer *)ptr);
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

    static void pinnedDestroy(cl::Buffer *buf, void *ptr)
    {
        getQueue().enqueueUnmapMemObject(*buf, (void *)ptr);
        destroy(buf);
    }

    void pinnedGarbageCollect()
    {
        int n = getActiveDeviceId();
        for(auto &iter : pinned_maps[n]) {
            if ((iter.second).info.is_free) {
                pinnedDestroy(iter.second.buf, iter.first);
            }
        }

        pinned_iter memory_curr = pinned_maps[n].begin();
        pinned_iter memory_end  = pinned_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.info.is_free) {
                memory_curr = pinned_maps[n].erase(memory_curr);
            } else {
                ++memory_curr;
            }
        }

    }

    void *pinnedBufferAlloc(const size_t &bytes)
    {
        void *ptr = NULL;
        int n = getActiveDeviceId();
        // Allocate the higher megabyte. Overhead of creating pinned memory is
        // more so we want more resuable memory.
        size_t alloc_bytes = divup(bytes, 1048576) * 1048576;

        if (bytes > 0) {
            cl::Buffer *buf = NULL;

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (pinned_maps[n].size() >= MAX_BUFFERS || pinned_used_bytes >= MAX_BYTES) {
                pinnedGarbageCollect();
            }

            for(pinned_iter iter = pinned_maps[n].begin();
                iter != pinned_maps[n].end(); ++iter) {

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
            mem_info info = {false, false, alloc_bytes};
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
            pinnedDestroy(iter->second.buf, ptr); // Free it because we are not sure what the size is
            pinned_maps[n].erase(iter);
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

#define INSTANTIATE(T)                                  \
    template T* memAlloc(const size_t &elements);       \
    template void memFree(T* ptr);                      \
    template void memPop(const T* ptr);                 \
    template void memPush(const T* ptr);                \
    template T* pinnedAlloc(const size_t &elements);    \
    template void pinnedFree(T* ptr);                   \

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
