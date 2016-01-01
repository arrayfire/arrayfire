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
#include <mutex>
#include <iostream>
#include <iomanip>
#include <string>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
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

typedef struct
{
    bool mngr_lock; // True if locked by memory manager, false if free
    bool user_lock; // True if locked by user, false if free
    size_t bytes;
} mem_info;

static size_t used_bytes = 0;
static size_t used_buffers = 0;
static size_t total_bytes = 0;
typedef std::map<void *, mem_info> mem_t;
typedef mem_t::iterator mem_iter;

mem_t memory_map;
std::mutex memory_map_mutex;

template<typename T>
void freeWrapper(T *ptr)
{
    free((void *)ptr);
}

void garbageCollect()
{
    for(mem_iter iter = memory_map.begin();
        iter != memory_map.end(); ++iter) {

        if (!(iter->second).mngr_lock) {

            if (!(iter->second).user_lock) {
                freeWrapper(iter->first);
                total_bytes -= iter->second.bytes;
            }
        }
    }

    mem_iter memory_curr = memory_map.begin();
    mem_iter memory_end  = memory_map.end();

    while(memory_curr != memory_end) {
        if (memory_curr->second.mngr_lock || memory_curr->second.user_lock) {
            ++memory_curr;
        } else {
            memory_map.erase(memory_curr++);
        }
    }
}

void printMemInfo(const char *msg, const int device)
{
    std::cout << msg << std::endl;

    static const std::string head("|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |");
    static const std::string line(head.size(), '-');
    std::cout << line << std::endl << head << std::endl << line << std::endl;

    for(mem_iter iter = memory_map.begin();
        iter != memory_map.end(); ++iter) {

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

template<typename T>
T* memAlloc(const size_t &elements)
{
    managerInit();

    T* ptr = NULL;
    size_t alloc_bytes = divup(sizeof(T) * elements, memory_resolution) * memory_resolution;

    if (elements > 0) {
        std::lock_guard<std::mutex> lock(memory_map_mutex);

        // FIXME: Add better checks for garbage collection
        // Perhaps look at total memory available as a metric
        if (memory_map.size() > MAX_BUFFERS ||
            used_bytes >= MAX_BYTES) {

            garbageCollect();
        }

        for(mem_iter iter = memory_map.begin();
            iter != memory_map.end(); ++iter) {

            mem_info info = iter->second;

            if (!info.mngr_lock &&
                !info.user_lock &&
                 info.bytes == alloc_bytes) {

                iter->second.mngr_lock = true;
                used_bytes += alloc_bytes;
                used_buffers++;
                return (T *)iter->first;
            }
        }

        // Perform garbage collection if memory can not be allocated
        ptr = (T *)malloc(alloc_bytes);

        if (ptr == NULL) {
            AF_ERROR("Can not allocate memory", AF_ERR_NO_MEM);
        }

        mem_info info = {true, false, alloc_bytes};
        memory_map[ptr] = info;

        used_bytes += alloc_bytes;
        used_buffers++;
        total_bytes += alloc_bytes;
    }
    return ptr;
}

template<typename T>
void memFreeLocked(T *ptr, bool freeLocked)
{
    std::lock_guard<std::mutex> lock(memory_map_mutex);

    mem_iter iter = memory_map.find((void *)ptr);

    if (iter != memory_map.end()) {

        iter->second.mngr_lock = false;
        if ((iter->second).user_lock && !freeLocked) return;

        iter->second.user_lock = false;
        used_bytes -= iter->second.bytes;
        used_buffers--;

    } else {
        freeWrapper(ptr); // Free it because we are not sure what the size is
    }
}

template<typename T>
void memFree(T *ptr)
{
    memFreeLocked(ptr, false);
}

template<typename T>
void memPop(const T *ptr)
{
    std::lock_guard<std::mutex> lock(memory_map_mutex);

    mem_iter iter = memory_map.find((void *)ptr);

    if (iter != memory_map.end()) {
        iter->second.user_lock = true;
    } else {
        mem_info info = { true,
                          true,
                          100 }; //This number is not relevant

        memory_map[(void *)ptr] = info;
    }
}

template<typename T>
void memPush(const T *ptr)
{
    std::lock_guard<std::mutex> lock(memory_map_mutex);
    mem_iter iter = memory_map.find((void *)ptr);
    if (iter != memory_map.end()) {
        iter->second.user_lock = false;
    }
}


void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes,  size_t *lock_buffers)
{
    getQueue().sync();
    if (alloc_bytes   ) *alloc_bytes   = total_bytes;
    if (alloc_buffers ) *alloc_buffers = memory_map.size();
    if (lock_bytes    ) *lock_bytes    = used_bytes;
    if (lock_buffers  ) *lock_buffers  = used_buffers;
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

#define INSTANTIATE(T)                                          \
    template T* memAlloc(const size_t &elements);               \
    template void memFree(T* ptr);                              \
    template void memFreeLocked(T* ptr, bool freeLocked);       \
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
INSTANTIATE(ushort)
INSTANTIATE(short )

}
