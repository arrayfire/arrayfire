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
#include <iostream>
#include <iomanip>
#include <string>
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
        bool mngr_lock;
        bool user_lock;
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

            if (!(iter->second).mngr_lock) {

                if (!(iter->second).user_lock) {
                    destroy(iter->first);
                    total_bytes[n] -= iter->second.bytes;
                }
            }
        }

        mem_iter memory_curr = memory_maps[n].begin();
        mem_iter memory_end  = memory_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.mngr_lock || memory_curr->second.user_lock) {
                ++memory_curr;
            } else {
                memory_maps[n].erase(memory_curr++);
            }
        }
    }

    void printMemInfo(const char *msg, const int device)
    {
        std::cout << msg << std::endl;
        std::cout << "Memory Map for Device: " << device << std::endl;

        static const std::string head("|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |");
        static const std::string line(head.size(), '-');
        std::cout << line << std::endl << head << std::endl << line << std::endl;

        for(mem_iter iter = memory_maps[device].begin();
            iter != memory_maps[device].end(); ++iter) {

            std::string status_mngr("Unknown");
            std::string status_user("Unknown");

            if(iter->second.mngr_lock)  status_mngr = "Yes";
            else                        status_mngr = " No";

            if(iter->second.user_lock)  status_user = "Yes";
            else                        status_user = " No";

            std::string unit = "KB";
            double size = (double)(iter->second.bytes) / 1024;
            if(size >= 1024) {
                size = size / 1024;
                unit = "MB";
            }

            std::cout << "|  " << std::right << std::setw(14) << iter->first << " "
                      << " | " << std::setw(7) << std::setprecision(4) << size << " " << unit
                      << " | " << std::setw(9) << status_mngr
                      << " | " << std::setw(9) << status_user
                      << " |"  << std::endl;
        }

        std::cout << line << std::endl;
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

                if (!info.mngr_lock &&
                    !info.user_lock &&
                     info.bytes == alloc_bytes) {

                    iter->second.mngr_lock = true;
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

            mem_info info = {true, false, alloc_bytes};
            memory_maps[n][ptr] = info;
            used_bytes[n] += alloc_bytes;
            used_buffers[n]++;
            total_bytes[n] += alloc_bytes;
        }
        return ptr;
    }

    void bufferFree(cl::Buffer *ptr)
    {
        bufferFreeUnlinked(ptr, false);
    }

    void bufferFreeUnlinked(cl::Buffer *ptr, bool free_unlinked)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {

            iter->second.mngr_lock = false;
            if ((iter->second).user_lock && !free_unlinked) return;

            iter->second.user_lock = false;

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
            iter->second.user_lock = true;
        } else {

            mem_info info = { true,
                              true,
                              100 }; //This number is not relevant

            memory_maps[n][ptr] = info;
        }
    }

    void bufferPush(cl::Buffer *ptr)
    {
        int n = getActiveDeviceId();
        mem_iter iter = memory_maps[n].find(ptr);

        if (iter != memory_maps[n].end()) {
            iter->second.user_lock = false;
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
        return bufferFreeUnlinked((cl::Buffer *)ptr, false);
    }

    template<typename T>
    void memFreeUnlinked(T *ptr, bool free_unlinked)
    {
        return bufferFreeUnlinked((cl::Buffer *)ptr, free_unlinked);
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
            if (!(iter.second).info.mngr_lock) {
                pinnedDestroy(iter.second.buf, iter.first);
            }
        }

        pinned_iter memory_curr = pinned_maps[n].begin();
        pinned_iter memory_end  = pinned_maps[n].end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.info.mngr_lock) {
                ++memory_curr;
            } else {
                memory_curr = pinned_maps[n].erase(memory_curr);
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
                if (!info.mngr_lock && info.bytes == alloc_bytes) {
                    iter->second.info.mngr_lock = true;
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
            mem_info info = {true, false, alloc_bytes};
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
            iter->second.info.mngr_lock = false;
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

#define INSTANTIATE(T)                                          \
    template T* memAlloc(const size_t &elements);               \
    template void memFree(T* ptr);                              \
    template void memFreeUnlinked(T* ptr, bool free_unlinked);  \
    template void memPop(const T* ptr);                         \
    template void memPush(const T* ptr);                        \
    template T* pinnedAlloc(const size_t &elements);            \
    template void pinnedFree(T* ptr);                           \

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
}
