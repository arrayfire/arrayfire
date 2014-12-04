/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <err_cpu.hpp>
#include <types.hpp>
#include <map>
#include <dispatch.hpp>

namespace cpu
{
    typedef struct
    {
        bool is_free;
        size_t bytes;
    } mem_info;

    static size_t used_bytes = 0;
    typedef std::map<void *, mem_info> mem_t;
    typedef mem_t::iterator mem_iter;

    mem_t memory_map;

    template<typename T>
    void freeWrapper(T *ptr)
    {
        delete[] ptr;
    }

    void garbageCollect()
    {
        for(mem_iter iter = memory_map.begin(); iter != memory_map.end(); iter++) {
            if ((iter->second).is_free) freeWrapper(iter->first);
        }

        mem_iter memory_curr = memory_map.begin();
        mem_iter memory_end  = memory_map.end();

        while(memory_curr != memory_end) {
            if (memory_curr->second.is_free) {
                memory_map.erase(memory_curr++);
            } else {
                ++memory_curr;
            }
        }
    }

    template<typename T>
    T* memAlloc(const size_t &elements)
    {
        T* ptr = NULL;
        size_t alloc_bytes = divup(sizeof(T) * elements, 1024) * 1024;

        if (elements > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (memory_map.size() > 100 || used_bytes >= 100 * (1 << 20)) {
                garbageCollect();
            }

            for(mem_iter iter = memory_map.begin(); iter != memory_map.end(); iter++) {
                mem_info info = iter->second;
                if (info.is_free && info.bytes == alloc_bytes) {
                    iter->second.is_free = false;
                    used_bytes += alloc_bytes;
                    return (T *)iter->first;
                }
            }

            // Perform garbage collection if memory can not be allocated
            ptr = new T[elements];

            mem_info info = {false, alloc_bytes};
            memory_map[ptr] = info;

            used_bytes += alloc_bytes;
        }
        return ptr;
    }

    template<typename T>
    void memFree(T *ptr)
    {
        mem_iter iter = memory_map.find((void *)ptr);

        if (iter != memory_map.end()) {
            iter->second.is_free = true;
            used_bytes -= iter->second.bytes;
        } else {
            freeWrapper(ptr); // Free it because we are not sure what the size is
        }
    }

#define INSTANTIATE(T)                              \
    template T* memAlloc(const size_t &elements);   \
    template void memFree(T* ptr);                  \

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
