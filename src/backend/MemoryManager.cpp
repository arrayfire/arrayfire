/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "MemoryManager.hpp"
#include "dispatch.hpp"
#include "err_common.hpp"
#include "util.hpp"

namespace common
{

MemoryManager::MemoryManager(int num_devices, unsigned MAX_BUFFERS, bool debug):
    mem_step_size(1024),
    max_buffers(MAX_BUFFERS),
    memory(num_devices),
    debug_mode(debug)
{
    lock_guard_t lock(this->memory_mutex);

    for (int n = 0; n < num_devices; n++) {
        // Calling getMaxMemorySize() here calls the virtual function that returns 0
        // Call it from outside the constructor.
        memory[n].max_bytes     = ONE_GB;
        memory[n].total_bytes   = 0;
        memory[n].total_buffers = 0;
        memory[n].lock_bytes    = 0;
        memory[n].lock_buffers  = 0;
    }

    // Check for environment variables

    std::string env_var;

    // Debug mode
    env_var = getEnvVar("AF_MEM_DEBUG");
    if (!env_var.empty()) {
        this->debug_mode = env_var[0] != '0';
    }
    if (this->debug_mode) mem_step_size = 1;

    // Max Buffer count
    env_var = getEnvVar("AF_MAX_BUFFERS");
    if (!env_var.empty()) {
        this->max_buffers = std::max(1, std::stoi(env_var));
    }
}

void MemoryManager::setMaxMemorySize()
{
    for (unsigned n = 0; n < memory.size(); n++) {
        // Calls garbage collection when:
        // total_bytes > memsize * 0.75 when memsize <  4GB
        // total_bytes > memsize - 1 GB when memsize >= 4GB
        // If memsize returned 0, then use 1GB
        size_t memsize = this->getMaxMemorySize(n);
        memory[n].max_bytes = memsize == 0 ? ONE_GB : std::max(memsize * 0.75, (double)(memsize - ONE_GB));
    }
}

void MemoryManager::garbageCollect()
{
    if (this->debug_mode) return;

    lock_guard_t lock(this->memory_mutex);
    memory_info& current = this->getCurrentMemoryInfo();

    // Return if all buffers are locked
    if (current.total_buffers == current.lock_buffers) return;

    for (auto &kv : current.free_map) {
        size_t num_ptrs = kv.second.size();
        //Free memory by popping the last element
        for (int n = num_ptrs-1; n >= 0; n--) {
            this->nativeFree(kv.second[n]);
            current.total_bytes -= kv.first;
            current.total_buffers--;
            kv.second.pop_back();
        }
    }
}

void MemoryManager::unlock(void *ptr, bool user_unlock)
{
    lock_guard_t lock(this->memory_mutex);
    memory_info& current = this->getCurrentMemoryInfo();

    locked_iter iter = current.locked_map.find((void *)ptr);

    // Pointer not found in locked map
    if (iter == current.locked_map.end()) {
        // Probably came from user, just free it
        this->nativeFree(ptr);
        return;
    }

    if (user_unlock) {
        (iter->second).user_lock = false;
    } else {
        (iter->second).manager_lock = false;
    }

    // Return early if either one is locked
    if ((iter->second).user_lock || (iter->second).manager_lock) return;

    size_t bytes = iter->second.bytes;
    current.lock_bytes -= iter->second.bytes;
    current.lock_buffers--;

    current.locked_map.erase(iter);

    if (this->debug_mode) {
        // Just free memory in debug mode
        if ((iter->second).bytes > 0) {
            this->nativeFree(iter->first);
        }
    } else {
        // In regular mode, move buffer to free map
        free_iter fiter = current.free_map.find(bytes);
        if (fiter != current.free_map.end()) {
            // If found, push back
            fiter->second.push_back(ptr);
        } else {
            // If not found, create new vector for this size
            std::vector<void *> ptrs;
            ptrs.push_back(ptr);
            current.free_map[bytes] = ptrs;
        }
    }
}

void *MemoryManager::alloc(const size_t bytes, bool user_lock)
{
    lock_guard_t lock(this->memory_mutex);

    void *ptr = NULL;
    size_t alloc_bytes = this->debug_mode ? bytes : (divup(bytes, mem_step_size) * mem_step_size);

    if (bytes > 0) {
        memory_info& current = this->getCurrentMemoryInfo();

        // There is no memory cache in debug mode
        if (!this->debug_mode) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (current.lock_bytes >= current.max_bytes ||
                current.total_buffers >= this->max_buffers) {
                this->garbageCollect();
            }

            free_iter iter = current.free_map.find(alloc_bytes);

            if (iter != current.free_map.end() && !iter->second.empty()) {
                ptr = iter->second.back();
                iter->second.pop_back();
            }

        }

        // Only comes here if buffer size not found or in debug mode
        if (ptr == NULL) {
            // Perform garbage collection if memory can not be allocated
            try {
                ptr = this->nativeAlloc(alloc_bytes);
            } catch (AfError &ex) {
                // If out of memory, run garbage collect and try again
                if (ex.getError() != AF_ERR_NO_MEM) throw;
                this->garbageCollect();
                ptr = this->nativeAlloc(alloc_bytes);
            }
            // Increment these two only when it succeeds to come here.
            current.total_bytes += alloc_bytes;
            current.total_buffers += 1;
        }


        locked_info info = {true, user_lock, alloc_bytes};
        current.locked_map[ptr] = info;
        current.lock_bytes += alloc_bytes;
        current.lock_buffers++;
    }
    return ptr;
}

void MemoryManager::userLock(const void *ptr)
{
    memory_info& current = this->getCurrentMemoryInfo();

    lock_guard_t lock(this->memory_mutex);

    locked_iter iter = current.locked_map.find(const_cast<void *>(ptr));

    if (iter != current.locked_map.end()) {
        iter->second.user_lock = true;
    } else {
        locked_info info = {false,
                            true,
                            100}; //This number is not relevant

        current.locked_map[(void *)ptr] = info;
    }
}

void MemoryManager::userUnlock(const void *ptr)
{
    this->unlock(const_cast<void *>(ptr), true);
}

size_t MemoryManager::getMemStepSize()
{
    lock_guard_t lock(this->memory_mutex);
    return this->mem_step_size;
}

void MemoryManager::setMemStepSize(size_t new_step_size)
{
    lock_guard_t lock(this->memory_mutex);
    this->mem_step_size = new_step_size;
}

size_t MemoryManager::getMaxBytes()
{
    lock_guard_t lock(this->memory_mutex);
    return this->getCurrentMemoryInfo().max_bytes;
}

void MemoryManager::printInfo(const char *msg, const int device)
{
    lock_guard_t lock(this->memory_mutex);
    memory_info& current = this->getCurrentMemoryInfo();

    std::cout << msg << std::endl;

    static const std::string head("|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |");
    static const std::string line(head.size(), '-');
    std::cout << line << std::endl << head << std::endl << line << std::endl;

    for(auto& kv : current.locked_map) {
        std::string status_mngr("Yes");
        std::string status_user("Unknown");
        if(kv.second.user_lock)     status_user = "Yes";
        else                        status_user = " No";

        std::string unit = "KB";
        double size = (double)(kv.second.bytes) / 1024;
        if(size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        std::cout << " |  " << std::right << std::setw(14) << kv.first << " "
                  << " | " << std::setw(7) << std::setprecision(4) << size << " " << unit
                  << " | " << std::setw(9) << status_mngr
                  << " | " << std::setw(9) << status_user
                  << " |"  << std::endl;
    }

    for(auto &kv : current.free_map) {

        std::string status_mngr("No");
        std::string status_user("No");

        std::string unit = "KB";
        double size = (double)(kv.first) / 1024;
        if(size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        for (auto &ptr : kv.second) {
            std::cout << " |  " << std::right << std::setw(14) << ptr << " "
                      << " | " << std::setw(7) << std::setprecision(4) << size << " " << unit
                      << " | " << std::setw(9) << status_mngr
                      << " | " << std::setw(9) << status_user
                      << " |"  << std::endl;
        }
    }

    std::cout << line << std::endl;
}

void MemoryManager::bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                               size_t *lock_bytes,  size_t *lock_buffers)
{
    lock_guard_t lock(this->memory_mutex);
    memory_info current = this->getCurrentMemoryInfo();
    if (alloc_bytes   ) *alloc_bytes   = current.total_bytes;
    if (alloc_buffers ) *alloc_buffers = current.total_buffers;
    if (lock_bytes    ) *lock_bytes    = current.lock_bytes;
    if (lock_buffers  ) *lock_buffers  = current.lock_buffers;
}

unsigned MemoryManager::getMaxBuffers()
{
    return this->max_buffers;
}

}
