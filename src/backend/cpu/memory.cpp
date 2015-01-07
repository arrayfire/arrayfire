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
#include <cstdlib>

namespace cpu
{
    void garbageCollect();

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
            garbageCollect();
        }
    };

    bool Manager::initialized = false;

    static void managerInit()
    {
        if(Manager::initialized == false)
            static Manager pm = Manager();
    }

    static const unsigned MAX_BUFFERS = 100;
    static const unsigned MAX_BYTES = 100 * (1 << 20);
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
        free((void *)ptr);
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
        managerInit();

        T* ptr = NULL;
        size_t alloc_bytes = divup(sizeof(T) * elements, 1024) * 1024;

        if (elements > 0) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (memory_map.size() > MAX_BUFFERS || used_bytes >= MAX_BYTES) {
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
            ptr = (T *)malloc(alloc_bytes);

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

    template<typename T>
    T* pinnedAlloc(const size_t &elements)
    {
        return memAlloc<T>(elements);
    }

    template<typename T>
    void pinnedFree(T* ptr)
    {
        memFree<T>(ptr);
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
