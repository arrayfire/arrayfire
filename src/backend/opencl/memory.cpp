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

    cl::Buffer *memAlloc(const size_t &bytes)
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

    void memFree(cl::Buffer *ptr)
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
}
