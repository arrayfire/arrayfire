/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>
#include <common/Logger.hpp>
#include <common/MemoryManager.hpp>
#include <memoryapi.hpp>
#include <af/event.h>
#include <af/memory.h>

#include <string>
#include <vector>

using af::buffer_info;
using af::event;
using std::max;
using std::stoi;
using std::string;
using std::vector;

using spdlog::logger;

namespace common {

memory::memory_info &MemoryManager::getCurrentMemoryInfo() {
    return memory[this->getActiveDeviceId()];
}

void MemoryManager::cleanDeviceMemoryManager(int device) {
    if (this->debug_mode) return;

    // This vector is used to store the pointers which will be deleted by
    // the memory manager. We are using this to avoid calling free while
    // the lock is being held because the CPU backend calls sync.
    vector<af_buffer_info> free_ptrs;
    size_t bytes_freed           = 0;
    memory::memory_info &current = memory[device];
    {
        lock_guard_t lock(this->memory_mutex);
        // Return if all buffers are locked
        if (current.total_buffers == current.lock_buffers) return;
        free_ptrs.reserve(32);

        for (auto &kv : current.free_map) {
            size_t num_ptrs = kv.second.size();
            // Free memory by pushing the last element into the free_ptrs
            // vector which will be freed once outside of the lock
            for (auto &pair : kv.second) { free_ptrs.emplace_back(pair); }
            current.total_bytes -= num_ptrs * kv.first;
            bytes_freed += num_ptrs * kv.first;
            current.total_buffers -= num_ptrs;
        }
        current.free_map.clear();
    }

    AF_TRACE("GC: Clearing {} buffers {}", free_ptrs.size(),
             bytesToString(bytes_freed));
    // Free memory outside of the lock
    for (auto &pair : free_ptrs) {
        void *ptr;
        af_unlock_buffer_info_ptr(&ptr, pair);
        this->nativeFree(ptr);
        // Release resources
        af_delete_buffer_info(pair);
    }
}

MemoryManager::MemoryManager(int num_devices, unsigned max_buffers, bool debug)
    : mem_step_size(1024)
    , max_buffers(max_buffers)
    , memory(num_devices)
    , debug_mode(debug) {
    // Check for environment variables

    // Debug mode
    string env_var = getEnvVar("AF_MEM_DEBUG");
    if (!env_var.empty()) this->debug_mode = env_var[0] != '0';
    if (this->debug_mode) mem_step_size = 1;

    // Max Buffer count
    env_var = getEnvVar("AF_MAX_BUFFERS");
    if (!env_var.empty()) this->max_buffers = max(1, stoi(env_var));
}

void MemoryManager::initialize() { this->setMaxMemorySize(); }

void MemoryManager::shutdown() { garbageCollect(); }

void MemoryManager::addMemoryManagement(int device) {
    // If there is a memory manager allocated for this device id, we might
    // as well use it and the buffers allocated for it
    if (static_cast<size_t>(device) < memory.size()) return;

    // Assuming, device need not be always the next device Lets resize to
    // current_size + device + 1 +1 is to account for device being 0-based
    // index of devices
    memory.resize(memory.size() + device + 1);
}

void MemoryManager::removeMemoryManagement(int device) {
    if ((size_t)device >= memory.size())
        AF_ERROR("No matching device found", AF_ERR_ARG);

    // Do garbage collection for the device and leave the memory::memory_info
    // struct from the memory vector intact
    cleanDeviceMemoryManager(device);
}

void MemoryManager::setMaxMemorySize() {
    for (unsigned n = 0; n < memory.size(); n++) {
        // Calls garbage collection when: total_bytes > memsize * 0.75 when
        // memsize < 4GB total_bytes > memsize - 1 GB when memsize >= 4GB If
        // memsize returned 0, then use 1GB
        size_t memsize = this->getMaxMemorySize(n);
        memory[n].max_bytes =
            memsize == 0 ? ONE_GB
                         : max(memsize * 0.75, (double)(memsize - ONE_GB));
    }
}

af_buffer_info MemoryManager::alloc(const size_t bytes, bool user_lock) {
    af_event event = detail::createAndMarkEvent();
    af_buffer_info bufferInfo;
    af_create_buffer_info(&bufferInfo, nullptr, event);
    // buffer_info pairObj(nullptr, e.get());
    size_t alloc_bytes = this->debug_mode
                             ? bytes
                             : (divup(bytes, mem_step_size) * mem_step_size);

    if (bytes > 0) {
        memory::memory_info &current = this->getCurrentMemoryInfo();
        memory::locked_info info     = {!user_lock, user_lock, alloc_bytes};

        // There is no memory cache in debug mode
        if (!this->debug_mode) {
            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (this->checkMemoryLimit()) { this->garbageCollect(); }

            lock_guard_t lock(this->memory_mutex);
            memory::free_iter iter = current.free_map.find(alloc_bytes);

            if (iter != current.free_map.end() && !iter->second.empty()) {
                // Release existing buffer info and underlying event
                af_delete_buffer_info(bufferInfo);
                // Set to existing in from free map
                bufferInfo = iter->second.back();
                af_buffer_info_get_event(&event, bufferInfo);
                iter->second.pop_back();
                current.locked_map[getBufferInfo(bufferInfo).ptr] = info;
                current.lock_bytes += alloc_bytes;
                current.lock_buffers++;
            }
        }

        void *ptr = getBufferInfo(bufferInfo).ptr;
        // Only comes here if buffer size not found or in debug mode
        if (ptr == nullptr) {
            // Perform garbage collection if memory can not be allocated
            try {
                ptr = this->nativeAlloc(alloc_bytes);
                af_buffer_info_set_ptr(bufferInfo, ptr);
            } catch (const AfError &ex) {
                // If out of memory, run garbage collect and try again
                if (ex.getError() != AF_ERR_NO_MEM) throw;
                this->garbageCollect();
                ptr = this->nativeAlloc(alloc_bytes);
                af_buffer_info_set_ptr(bufferInfo, ptr);
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
    return bufferInfo;
}

size_t MemoryManager::allocated(void *ptr) {
    if (!ptr) return 0;
    memory::memory_info &current = this->getCurrentMemoryInfo();
    memory::locked_iter iter     = current.locked_map.find((void *)ptr);
    if (iter == current.locked_map.end()) return 0;
    return (iter->second).bytes;
}

void MemoryManager::unlock(void *ptr, af_event eventHandle, bool user_unlock) {
    // Shortcut for empty arrays
    if (!ptr) {
        af_release_event(eventHandle);
        return;
    }

    // Frees the pointer outside the lock.
    memory::uptr_t freed_ptr(nullptr, [this](void *p) { this->nativeFree(p); });
    {
        lock_guard_t lock(this->memory_mutex);
        memory::memory_info &current = this->getCurrentMemoryInfo();

        memory::locked_iter iter = current.locked_map.find((void *)ptr);

        // Pointer not found in locked map
        if (iter == current.locked_map.end()) {
            // Probably came from user, just free it
            freed_ptr.reset(ptr);
            af_release_event(eventHandle);
            return;
        }

        if (user_unlock) {
            (iter->second).user_lock = false;
        } else {
            (iter->second).manager_lock = false;
        }

        // Return early if either one is locked
        if ((iter->second).user_lock || (iter->second).manager_lock) {
            af_release_event(eventHandle);
            return;
        }

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
            af_release_event(eventHandle);
        } else {
            af_buffer_info info;
            af_create_buffer_info(&info, ptr, eventHandle);
            current.free_map[bytes].emplace_back(info);
        }
        current.locked_map.erase(iter);
    }
}

void MemoryManager::garbageCollect() {
    cleanDeviceMemoryManager(this->getActiveDeviceId());
}

void MemoryManager::printInfo(const char *msg, const int device) {
    const memory::memory_info &current = this->getCurrentMemoryInfo();

    printf("%s\n", msg);
    printf(
        "---------------------------------------------------------\n"
        "|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |\n"
        "---------------------------------------------------------\n");

    lock_guard_t lock(this->memory_mutex);
    for (auto &kv : current.locked_map) {
        const char *status_mngr = "Yes";
        const char *status_user = "Unknown";
        if (kv.second.user_lock)
            status_user = "Yes";
        else
            status_user = " No";

        const char *unit = "KB";
        double size      = (double)(kv.second.bytes) / 1024;
        if (size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        printf("|  %14p  |  %6.f %s | %9s | %9s |\n", kv.first, size, unit,
               status_mngr, status_user);
    }

    for (auto &kv : current.free_map) {
        const char *status_mngr = "No";
        const char *status_user = "No";

        const char *unit = "KB";
        double size      = (double)(kv.first) / 1024;
        if (size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        for (auto &pair : kv.second) {
            printf("|  %14p  |  %6.f %s | %9s | %9s |\n",
                   getBufferInfo(pair).ptr, size, unit, status_mngr,
                   status_user);
        }
    }

    printf("---------------------------------------------------------\n");
}

void MemoryManager::usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                              size_t *lock_bytes, size_t *lock_buffers) {
    const memory::memory_info &current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);
    if (alloc_bytes) *alloc_bytes = current.total_bytes;
    if (alloc_buffers) *alloc_buffers = current.total_buffers;
    if (lock_bytes) *lock_bytes = current.lock_bytes;
    if (lock_buffers) *lock_buffers = current.lock_buffers;
}

void MemoryManager::userLock(const void *ptr) {
    memory::memory_info &current = this->getCurrentMemoryInfo();

    lock_guard_t lock(this->memory_mutex);

    memory::locked_iter iter = current.locked_map.find(const_cast<void *>(ptr));
    if (iter != current.locked_map.end()) {
        iter->second.user_lock = true;
    } else {
        memory::locked_info info = {false, true,
                                    100};  // This number is not relevant

        current.locked_map[(void *)ptr] = info;
    }
}

void MemoryManager::userUnlock(const void *ptr) {
    this->unlock(const_cast<void *>(ptr), detail::createAndMarkEvent(), true);
}

bool MemoryManager::isUserLocked(const void *ptr) {
    memory::memory_info &current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);
    memory::locked_iter iter = current.locked_map.find(const_cast<void *>(ptr));
    if (iter != current.locked_map.end()) {
        return iter->second.user_lock;
    } else {
        return false;
    }
}

size_t MemoryManager::getMemStepSize() {
    lock_guard_t lock(this->memory_mutex);
    return this->mem_step_size;
}

size_t MemoryManager::getMaxBytes() {
    lock_guard_t lock(this->memory_mutex);
    return this->getCurrentMemoryInfo().max_bytes;
}

unsigned MemoryManager::getMaxBuffers() { return this->max_buffers; }

void MemoryManager::setMemStepSize(size_t new_step_size) {
    lock_guard_t lock(this->memory_mutex);
    this->mem_step_size = new_step_size;
}

bool MemoryManager::checkMemoryLimit() {
    const memory::memory_info &current = this->getCurrentMemoryInfo();
    return current.lock_bytes >= current.max_bytes ||
           current.total_buffers >= this->max_buffers;
}
}  // namespace common
