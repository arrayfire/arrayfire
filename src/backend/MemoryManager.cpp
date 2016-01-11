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
#include "MemoryManager.hpp"
#include "dispatch.hpp"
#include "err_common.hpp"
#include "util.hpp"

namespace common
{

MemoryManager::MemoryManager(int num_devices, unsigned MAX_BUFFERS, unsigned MAX_BYTES, bool debug):
    mem_step_size(1024),
    max_buffers(MAX_BUFFERS),
    max_bytes(MAX_BYTES),
    memory(num_devices),
    debug_mode(debug)
{
    std::string env_var = getEnvVar("AF_MEM_DEBUG");
    if (!env_var.empty()) {
        this->debug_mode = env_var[0] != '0';
    }
    if (this->debug_mode) mem_step_size = 1;
}

void MemoryManager::garbageCollect()
{
    if (this->debug_mode) return;

    memory_info& current = this->getCurrentMemoryInfo();

    for(buffer_iter iter = current.map.begin();
        iter != current.map.end(); ++iter) {

        if (!(iter->second).manager_lock) {

            if (!(iter->second).user_lock) {
                if ((iter->second).bytes > 0) {
                    this->nativeFree(iter->first);
                }
                current.total_bytes -= iter->second.bytes;
            }
        }
    }

    buffer_iter memory_curr = current.map.begin();
    buffer_iter memory_end  = current.map.end();

    while(memory_curr != memory_end) {
        if (memory_curr->second.manager_lock || memory_curr->second.user_lock) {
            ++memory_curr;
        } else {
            current.map.erase(memory_curr++);
        }
    }
}

void MemoryManager::unlock(void *ptr, bool user_unlock)
{
    memory_info& current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);

    buffer_iter iter = current.map.find((void *)ptr);

    if (iter != current.map.end()) {

        iter->second.manager_lock = false;
        if ((iter->second).user_lock && !user_unlock) return;

        iter->second.user_lock = false;
        current.lock_bytes -= iter->second.bytes;
        current.lock_buffers--;

        if (this->debug_mode) {
            if ((iter->second).bytes > 0) {
                this->nativeFree(iter->first);
            }
        }

    } else {
        this->nativeFree(ptr); // Free it because we are not sure what the size is
    }
}

void *MemoryManager::alloc(const size_t bytes)
{
    memory_info& current = this->getCurrentMemoryInfo();

    void *ptr = NULL;
    size_t alloc_bytes = this->debug_mode ? bytes : (divup(bytes, mem_step_size) * mem_step_size);

    if (bytes > 0) {

        lock_guard_t lock(this->memory_mutex);

        // There is no memory cache in debug mode
        if (!this->debug_mode) {

            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (current.map.size() > this->max_buffers ||
                current.lock_bytes >= this->max_bytes) {

                this->garbageCollect();
            }

            for(buffer_iter iter = current.map.begin();
                iter != current.map.end(); ++iter) {

                buffer_info info = iter->second;

                if (!info.manager_lock &&
                    !info.user_lock &&
                    info.bytes == alloc_bytes) {

                    iter->second.manager_lock = true;
                    current.lock_bytes += alloc_bytes;
                    current.lock_buffers++;
                    return iter->first;
                }
            }
        }

        // Perform garbage collection if memory can not be allocated
        ptr = this->nativeAlloc(alloc_bytes);

        if (!ptr) {
            this->garbageCollect();
            ptr = this->nativeAlloc(alloc_bytes);
            if (!ptr) AF_ERROR("Can not allocate memory", AF_ERR_NO_MEM);
        }

        buffer_info info = {true, false, alloc_bytes};
        current.map[ptr] = info;

        current.lock_bytes += alloc_bytes;
        current.lock_buffers++;
        current.total_bytes += alloc_bytes;
    }
    return ptr;
}

void MemoryManager::userLock(const void *ptr)
{
    memory_info& current = this->getCurrentMemoryInfo();

    lock_guard_t lock(this->memory_mutex);

    buffer_iter iter = current.map.find(const_cast<void *>(ptr));

    if (iter != current.map.end()) {
        iter->second.user_lock = true;
    } else {
        buffer_info info = { true,
                          true,
                          100 }; //This number is not relevant

        current.map[(void *)ptr] = info;
    }
}

void MemoryManager::userUnlock(const void *ptr)
{
    memory_info& current = this->getCurrentMemoryInfo();

    lock_guard_t lock(this->memory_mutex);

    buffer_iter iter = current.map.find((void *)ptr);
    if (iter != current.map.end()) {
        iter->second.user_lock = false;
        if (this->debug_mode) {
            if ((iter->second).bytes > 0) {
                this->nativeFree(iter->first);
            }
        }
    }
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

void MemoryManager::printInfo(const char *msg, const int device)
{
    lock_guard_t lock(this->memory_mutex);
    memory_info& current = this->getCurrentMemoryInfo();

    std::cout << msg << std::endl;

    static const std::string head("|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |");
    static const std::string line(head.size(), '-');
    std::cout << line << std::endl << head << std::endl << line << std::endl;

    for(buffer_iter iter = current.map.begin();
        iter != current.map.end(); ++iter) {

        std::string status_mngr("Unknown");
        std::string status_user("Unknown");

        if(iter->second.manager_lock)  status_mngr = "Yes";
        else                           status_mngr = " No";

        if(iter->second.user_lock)     status_user = "Yes";
        else                           status_user = " No";

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

void MemoryManager::bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                               size_t *lock_bytes,  size_t *lock_buffers)
{
    memory_info current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);
    if (alloc_bytes   ) *alloc_bytes   = current.total_bytes;
    if (alloc_buffers ) *alloc_buffers = current.map.size();
    if (lock_bytes    ) *lock_bytes    = current.lock_bytes;
    if (lock_buffers  ) *lock_buffers  = current.lock_buffers;
}
}
