/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <common/util.hpp>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace common
{
using mutex_t      = std::recursive_mutex;
using lock_guard_t = std::lock_guard<mutex_t>;

const unsigned MAX_BUFFERS   = 1000;
const size_t ONE_GB = 1 << 30;

template<typename T>
class MemoryManager
{
    typedef struct
    {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    } locked_info;

    using locked_t    = typename std::unordered_map<void *, locked_info>;
    using locked_iter = typename locked_t::iterator;

    using free_t    = std::unordered_map<size_t, std::vector<void *> >;
    using free_iter = free_t::iterator;

    using uptr_t = std::unique_ptr<void, std::function<void(void*)>>;

    typedef struct memory_info
    {
        locked_t locked_map;
        free_t   free_map;

        size_t lock_bytes;
        size_t lock_buffers;
        size_t total_bytes;
        size_t total_buffers;
        size_t max_bytes;

        memory_info()
        {
            // Calling getMaxMemorySize() here calls the virtual function that returns 0
            // Call it from outside the constructor.
            max_bytes     = ONE_GB;
            total_bytes   = 0;
            total_buffers = 0;
            lock_bytes    = 0;
            lock_buffers  = 0;
        }
    } memory_info;

    size_t mem_step_size;
    unsigned max_buffers;
    std::vector<memory_info> memory;
    bool debug_mode;

    memory_info& getCurrentMemoryInfo()
    {
        return memory[this->getActiveDeviceId()];
    }

    inline int getActiveDeviceId()
    {
        return static_cast<T*>(this)->getActiveDeviceId();
    }

    inline size_t getMaxMemorySize(int id)
    {
        return static_cast<T*>(this)->getMaxMemorySize(id);
    }

    void cleanDeviceMemoryManager(int device)
    {
        if (this->debug_mode) return;

        // This vector is used to store the pointers which will be deleted by
        // the memory manager. We are using this to avoid calling free while
        // the lock is being held becasue the CPU backend calls sync.
        std::vector<void*> free_ptrs;
        memory_info& current = memory[device];
        {
            lock_guard_t lock(this->memory_mutex);
            // Return if all buffers are locked
            if (current.total_buffers == current.lock_buffers) return;
            free_ptrs.reserve(32);

            for (auto &kv : current.free_map) {
                size_t num_ptrs = kv.second.size();
                // Free memory by pushing the last element into the free_ptrs
                // vector which will be freed once outside of the lock
                for(auto p : kv.second) {
                    free_ptrs.push_back(p);
                }
                current.total_bytes -= num_ptrs * kv.first;
                current.total_buffers -= num_ptrs;
            }
            current.free_map.clear();
        }
        // Free memory outside of the lock
        for(auto ptr : free_ptrs) {
            this->nativeFree(ptr);
        }
    }

    public:
    MemoryManager(int num_devices, unsigned max_buffers, bool debug)
        : mem_step_size(1024), max_buffers(max_buffers), memory(num_devices), debug_mode(debug)
    {
        // Check for environment variables

        // Debug mode
        std::string env_var = getEnvVar("AF_MEM_DEBUG");
        if (!env_var.empty()) this->debug_mode = env_var[0] != '0';
        if (this->debug_mode) mem_step_size = 1;

        // Max Buffer count
        env_var = getEnvVar("AF_MAX_BUFFERS");
        if (!env_var.empty()) this->max_buffers = std::max(1, std::stoi(env_var));
    }

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void addMemoryManagement(int device)
    {
        // If there is a memory manager allocated for
        // this device id, we might as well use it and the
        // buffers allocated for it
        if (static_cast<size_t>(device) < memory.size())
            return;

        // Assuming, device need not be always the next device
        // Lets resize to current_size + device + 1
        // +1 is to account for device being 0-based index of devices
        memory.resize(memory.size()+device+1);
    }

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void removeMemoryManagement(int device)
    {
        if ((size_t)device>=memory.size())
            AF_ERROR("No matching device found", AF_ERR_ARG);

        // Do garbage collection for the device and leave
        // the memory_info struct from the memory vector intact
        cleanDeviceMemoryManager(device);
    }

    void setMaxMemorySize()
    {
        for (unsigned n = 0; n < memory.size(); n++) {
            // Calls garbage collection when:
            // total_bytes > memsize * 0.75 when memsize <  4GB
            // total_bytes > memsize - 1 GB when memsize >= 4GB
            // If memsize returned 0, then use 1GB
            size_t memsize = this->getMaxMemorySize(n);
            memory[n].max_bytes = memsize == 0 ? ONE_GB :
                std::max(memsize * 0.75, (double)(memsize - ONE_GB));
        }
    }

    void *alloc(const size_t bytes, bool user_lock)
    {
        void *ptr = nullptr;
        size_t alloc_bytes =
          this->debug_mode ? bytes :
                             (divup(bytes, mem_step_size) * mem_step_size);

        if (bytes > 0) {
            memory_info& current = this->getCurrentMemoryInfo();
            locked_info info = {!user_lock, user_lock, alloc_bytes};

            // There is no memory cache in debug mode
            if (!this->debug_mode) {

                // FIXME: Add better checks for garbage collection
                // Perhaps look at total memory available as a metric
                if (this->checkMemoryLimit()) {
                    this->garbageCollect();
                }

                lock_guard_t lock(this->memory_mutex);
                free_iter iter = current.free_map.find(alloc_bytes);

                if (iter != current.free_map.end() && !iter->second.empty()) {
                    ptr = iter->second.back();
                    iter->second.pop_back();
                    current.locked_map[ptr] = info;
                    current.lock_bytes += alloc_bytes;
                    current.lock_buffers++;
                }
            }

            // Only comes here if buffer size not found or in debug mode
            if (ptr == nullptr) {
                // Perform garbage collection if memory can not be allocated
                try {
                    ptr = this->nativeAlloc(alloc_bytes);
                } catch (const AfError &ex) {
                    // If out of memory, run garbage collect and try again
                    if (ex.getError() != AF_ERR_NO_MEM) throw;
                    this->garbageCollect();
                    ptr = this->nativeAlloc(alloc_bytes);
                }

                lock_guard_t lock(this->memory_mutex);
                // Increment these two only when it succeeds to come here.
                current.total_bytes += alloc_bytes;
                current.total_buffers += 1;
                current.locked_map[ptr] = info;
                current.lock_bytes += alloc_bytes;
                current.lock_buffers++;
            }
        }
        return ptr;
    }

    size_t allocated(void *ptr)
    {
        if (!ptr) return 0;
        memory_info& current = this->getCurrentMemoryInfo();
        locked_iter iter = current.locked_map.find((void *)ptr);
        if (iter == current.locked_map.end()) return 0;
        return (iter->second).bytes;
    }

    void unlock(void *ptr, bool user_unlock)
    {
        // Shortcut for empty arrays
        if (!ptr) return;

        // Frees the pointer outside the lock.
        uptr_t freed_ptr(nullptr, [this](void* p) { this->nativeFree(p); });
        {
            lock_guard_t lock(this->memory_mutex);
            memory_info& current = this->getCurrentMemoryInfo();

            locked_iter iter = current.locked_map.find((void *)ptr);

            // Pointer not found in locked map
            if (iter == current.locked_map.end()) {
                // Probably came from user, just free it
                freed_ptr.reset(ptr);
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

            if (this->debug_mode) {
                // Just free memory in debug mode
                if ((iter->second).bytes > 0) {
                    freed_ptr.reset(iter->first);
                    current.total_buffers--;
                    current.total_bytes -= iter->second.bytes;
                }
            } else {
                current.free_map[bytes].push_back(ptr);
            }
            current.locked_map.erase(iter);
        }
    }

    void garbageCollect()
    {
        cleanDeviceMemoryManager(this->getActiveDeviceId());
    }

    void printInfo(const char *msg, const int device)
    {
        const memory_info& current = this->getCurrentMemoryInfo();

        printf("%s\n", msg);
        printf("---------------------------------------------------------\n"
               "|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |\n"
               "---------------------------------------------------------\n");

        lock_guard_t lock(this->memory_mutex);
        for(auto& kv : current.locked_map) {
            const char* status_mngr = "Yes";
            const char* status_user = "Unknown";
            if(kv.second.user_lock)     status_user = "Yes";
            else                        status_user = " No";

            const char* unit = "KB";
            double size = (double)(kv.second.bytes) / 1024;
            if(size >= 1024) {
                size = size / 1024;
                unit = "MB";
            }

            printf("|  %14p  |  %6.f %s | %9s | %9s |\n",
                   kv.first, size, unit, status_mngr, status_user);
        }

        for(auto &kv : current.free_map) {

            const char* status_mngr = "No";
            const char* status_user = "No";

            const char* unit = "KB";
            double size = (double)(kv.first) / 1024;
            if(size >= 1024) {
                size = size / 1024;
                unit = "MB";
            }

            for (auto &ptr : kv.second) {
              printf("|  %14p  |  %6.f %s | %9s | %9s |\n",
                     ptr, size, unit, status_mngr, status_user);
            }
        }

        printf("---------------------------------------------------------\n");
    }

    void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                    size_t *lock_bytes,  size_t *lock_buffers)
    {
        const memory_info& current = this->getCurrentMemoryInfo();
        lock_guard_t lock(this->memory_mutex);
        if (alloc_bytes   ) *alloc_bytes   = current.total_bytes;
        if (alloc_buffers ) *alloc_buffers = current.total_buffers;
        if (lock_bytes    ) *lock_bytes    = current.lock_bytes;
        if (lock_buffers  ) *lock_buffers  = current.lock_buffers;
    }

    void userLock(const void *ptr)
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

    void userUnlock(const void *ptr)
    {
        this->unlock(const_cast<void *>(ptr), true);
    }

    bool isUserLocked(const void *ptr)
    {
        memory_info& current = this->getCurrentMemoryInfo();
        lock_guard_t lock(this->memory_mutex);
        locked_iter iter = current.locked_map.find(const_cast<void *>(ptr));
        if (iter != current.locked_map.end()) {
            return iter->second.user_lock;
        } else {
            return false;
        }
    }

    size_t getMemStepSize()
    {
        lock_guard_t lock(this->memory_mutex);
        return this->mem_step_size;
    }

    size_t getMaxBytes()
    {
        lock_guard_t lock(this->memory_mutex);
        return this->getCurrentMemoryInfo().max_bytes;
    }

    unsigned getMaxBuffers()
    {
        return this->max_buffers;
    }

    void setMemStepSize(size_t new_step_size)
    {
        lock_guard_t lock(this->memory_mutex);
        this->mem_step_size = new_step_size;
    }

    inline void *nativeAlloc(const size_t bytes)
    {
        return static_cast<T*>(this)->nativeAlloc(bytes);
    }

    inline void nativeFree(void *ptr)
    {
        static_cast<T*>(this)->nativeFree(ptr);
    }

    virtual ~MemoryManager() {}

    bool checkMemoryLimit()
    {
        const memory_info& current = this->getCurrentMemoryInfo();
        return current.lock_bytes >= current.max_bytes || current.total_buffers >= this->max_buffers;
    }

    protected:
    mutex_t memory_mutex;
};

}
