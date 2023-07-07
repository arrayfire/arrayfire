/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <memoryapi.hpp>
#include <af/event.h>
#include <af/memory.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using std::max;
using std::move;
using std::stoi;
using std::string;
using std::vector;

namespace arrayfire {
namespace common {

DefaultMemoryManager::memory_info &
DefaultMemoryManager::getCurrentMemoryInfo() {
    return memory[this->getActiveDeviceId()];
}

void DefaultMemoryManager::cleanDeviceMemoryManager(int device) {
    if (this->debug_mode) { return; }

    // This vector is used to store the pointers which will be deleted by
    // the memory manager. We are using this to avoid calling free while
    // the lock is being held because the CPU backend calls sync.
    vector<void *> free_ptrs;
    size_t bytes_freed                         = 0;
    DefaultMemoryManager::memory_info &current = memory[device];
    {
        lock_guard_t lock(this->memory_mutex);
        // Return if all buffers are locked
        if (current.total_buffers == current.lock_buffers) { return; }
        free_ptrs.reserve(current.free_map.size());

        for (auto &kv : current.free_map) {
            size_t num_ptrs = kv.second.size();
            // Free memory by pushing the last element into the free_ptrs
            // vector which will be freed once outside of the lock
            // for (auto ptr : kv.second) { free_ptrs.emplace_back(pair); }
            std::move(begin(kv.second), end(kv.second),
                      back_inserter(free_ptrs));
            current.total_bytes -= num_ptrs * kv.first;
            bytes_freed += num_ptrs * kv.first;
            current.total_buffers -= num_ptrs;
        }
        current.free_map.clear();
    }

    AF_TRACE("GC: Clearing {} buffers {}", free_ptrs.size(),
             bytesToString(bytes_freed));
    // Free memory outside of the lock
    for (auto *ptr : free_ptrs) { this->nativeFree(ptr); }
}

DefaultMemoryManager::DefaultMemoryManager(int num_devices,
                                           unsigned max_buffers, bool debug)
    : mem_step_size(1024)
    , max_buffers(max_buffers)
    , debug_mode(debug)
    , memory(num_devices) {
    // Check for environment variables

    // Debug mode
    string env_var = getEnvVar("AF_MEM_DEBUG");
    if (!env_var.empty()) { this->debug_mode = env_var[0] != '0'; }
    if (this->debug_mode) { mem_step_size = 1; }

    // Max Buffer count
    env_var = getEnvVar("AF_MAX_BUFFERS");
    if (!env_var.empty()) { this->max_buffers = max(1, stoi(env_var)); }
}

void DefaultMemoryManager::initialize() { this->setMaxMemorySize(); }

void DefaultMemoryManager::shutdown() { signalMemoryCleanup(); }

void DefaultMemoryManager::addMemoryManagement(int device) {
    // If there is a memory manager allocated for this device id, we might
    // as well use it and the buffers allocated for it
    if (static_cast<size_t>(device) < memory.size()) { return; }

    // Assuming, device need not be always the next device Lets resize to
    // current_size + device + 1 +1 is to account for device being 0-based
    // index of devices
    memory.resize(memory.size() + device + 1);
}

void DefaultMemoryManager::removeMemoryManagement(int device) {
    if (static_cast<size_t>(device) >= memory.size()) {
        AF_ERROR("No matching device found", AF_ERR_ARG);
    }

    // Do garbage collection for the device and leave the memory::memory_info
    // struct from the memory vector intact
    cleanDeviceMemoryManager(device);
}

void DefaultMemoryManager::setMaxMemorySize() {
    for (unsigned n = 0; n < memory.size(); n++) {
        // Calls garbage collection when: total_bytes > memsize * 0.75 when
        // memsize < 4GB total_bytes > memsize - 1 GB when memsize >= 4GB If
        // memsize returned 0, then use 1GB
        size_t memsize = this->getMaxMemorySize(static_cast<int>(n));
        memory[n].max_bytes =
            memsize == 0
                ? ONE_GB
                : max(memsize * 0.75, static_cast<double>(memsize - ONE_GB));
        AF_TRACE("memory[{}].max_bytes: {}", n,
                 bytesToString(memory[n].max_bytes));
    }
}

float DefaultMemoryManager::getMemoryPressure() {
    lock_guard_t lock(this->memory_mutex);
    memory_info &current = this->getCurrentMemoryInfo();
    if (current.lock_bytes > current.max_bytes ||
        current.lock_buffers > max_buffers) {
        return 1.0;
    } else {
        return 0.0;
    }
}

bool DefaultMemoryManager::jitTreeExceedsMemoryPressure(
    size_t jit_tree_buffer_bytes) {
    lock_guard_t lock(this->memory_mutex);
    memory_info &current = this->getCurrentMemoryInfo();
    if (current.lock_bytes > 0.25f * current.max_bytes) {
        /// Evaluate JIT if half of all locked buffers are locked by this JIT
        /// tree
        return jit_tree_buffer_bytes > current.lock_bytes * 0.5f;
    } else {
        /// Evaluate if this JIT Tree accounts for 10% of total memory on the
        /// device
        return jit_tree_buffer_bytes > 0.10f * current.max_bytes;
    }
}

void *DefaultMemoryManager::alloc(bool user_lock, const unsigned ndims,
                                  dim_t *dims, const unsigned element_size) {
    size_t bytes = element_size;
    for (unsigned i = 0; i < ndims; ++i) { bytes *= dims[i]; }

    void *ptr          = nullptr;
    size_t alloc_bytes = this->debug_mode
                             ? bytes
                             : (divup(bytes, mem_step_size) * mem_step_size);

    if (bytes > 0) {
        memory_info &current = this->getCurrentMemoryInfo();
        locked_info info     = {!user_lock, user_lock, alloc_bytes};

        // There is no memory cache in debug mode
        if (!this->debug_mode) {
            // FIXME: Add better checks for garbage collection
            // Perhaps look at total memory available as a metric
            if (current.lock_bytes >= current.max_bytes ||
                current.total_buffers >= this->max_buffers) {
                AF_TRACE(
                    "Running GC: current.lock_bytes({}) >= "
                    "current.max_bytes({}) || current.total_buffers({}) >= "
                    "this->max_buffers({})\n",
                    current.lock_bytes, current.max_bytes,
                    current.total_buffers, this->max_buffers);

                this->signalMemoryCleanup();
            }

            lock_guard_t lock(this->memory_mutex);
            auto free_buffer_iter = current.free_map.find(alloc_bytes);
            if (free_buffer_iter != current.free_map.end() &&
                !free_buffer_iter->second.empty()) {
                // Delete existing buffer info and underlying event
                // Set to existing in from free map
                vector<void *> &free_buffer_vector = free_buffer_iter->second;
                ptr                                = free_buffer_vector.back();
                free_buffer_vector.pop_back();
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
                if (ex.getError() != AF_ERR_NO_MEM) { throw; }
                this->signalMemoryCleanup();
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

size_t DefaultMemoryManager::allocated(void *ptr) {
    if (!ptr) { return 0; }
    memory_info &current = this->getCurrentMemoryInfo();
    auto locked_iter     = current.locked_map.find(ptr);
    if (locked_iter == current.locked_map.end()) { return 0; }
    return (locked_iter->second).bytes;
}

void DefaultMemoryManager::unlock(void *ptr, bool user_unlock) {
    // Shortcut for empty arrays
    if (!ptr) { return; }

    // Frees the pointer outside the lock.
    uptr_t freed_ptr(nullptr, [this](void *p) { this->nativeFree(p); });
    {
        lock_guard_t lock(this->memory_mutex);
        memory_info &current = this->getCurrentMemoryInfo();

        auto locked_buffer_iter = current.locked_map.find(ptr);
        if (locked_buffer_iter == current.locked_map.end()) {
            // Pointer not found in locked map
            // Probably came from user, just free it
            freed_ptr.reset(ptr);
            return;
        }
        locked_info &locked_buffer_info = locked_buffer_iter->second;
        void *locked_buffer_ptr         = locked_buffer_iter->first;

        if (user_unlock) {
            locked_buffer_info.user_lock = false;
        } else {
            locked_buffer_info.manager_lock = false;
        }

        // Return early if either one is locked
        if (locked_buffer_info.user_lock || locked_buffer_info.manager_lock) {
            return;
        }

        size_t bytes = locked_buffer_info.bytes;
        current.lock_bytes -= locked_buffer_info.bytes;
        current.lock_buffers--;

        if (this->debug_mode) {
            // Just free memory in debug mode
            if (locked_buffer_info.bytes > 0) {
                freed_ptr.reset(locked_buffer_ptr);
                current.total_buffers--;
                current.total_bytes -= locked_buffer_info.bytes;
            }
        } else {
            current.free_map[bytes].emplace_back(ptr);
        }
        current.locked_map.erase(locked_buffer_iter);
    }
}

void DefaultMemoryManager::signalMemoryCleanup() {
    cleanDeviceMemoryManager(this->getActiveDeviceId());
}

void DefaultMemoryManager::printInfo(const char *msg, const int device) {
    UNUSED(device);
    const memory_info &current = this->getCurrentMemoryInfo();

    printf("%s\n", msg);
    printf(
        "---------------------------------------------------------\n"
        "|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |\n"
        "---------------------------------------------------------\n");

    lock_guard_t lock(this->memory_mutex);
    for (const auto &kv : current.locked_map) {
        const char *status_mngr = "Yes";
        const char *status_user = "Unknown";
        if (kv.second.user_lock) {
            status_user = "Yes";
        } else {
            status_user = " No";
        }

        const char *unit = "KB";
        double size      = static_cast<double>(kv.second.bytes) / 1024;
        if (size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        printf("|  %14p  |  %6.f %s | %9s | %9s |\n", kv.first, size, unit,
               status_mngr, status_user);
    }

    for (const auto &kv : current.free_map) {
        const char *status_mngr = "No";
        const char *status_user = "No";

        const char *unit = "KB";
        double size      = static_cast<double>(kv.first) / 1024;
        if (size >= 1024) {
            size = size / 1024;
            unit = "MB";
        }

        for (const auto &ptr : kv.second) {
            printf("|  %14p  |  %6.f %s | %9s | %9s |\n", ptr, size, unit,
                   status_mngr, status_user);
        }
    }

    printf("---------------------------------------------------------\n");
}

void DefaultMemoryManager::usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                                     size_t *lock_bytes, size_t *lock_buffers) {
    const memory_info &current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);
    if (alloc_bytes) { *alloc_bytes = current.total_bytes; }
    if (alloc_buffers) { *alloc_buffers = current.total_buffers; }
    if (lock_bytes) { *lock_bytes = current.lock_bytes; }
    if (lock_buffers) { *lock_buffers = current.lock_buffers; }
}

void DefaultMemoryManager::userLock(const void *ptr) {
    memory_info &current = this->getCurrentMemoryInfo();

    lock_guard_t lock(this->memory_mutex);

    auto locked_iter = current.locked_map.find(const_cast<void *>(ptr));
    if (locked_iter != current.locked_map.end()) {
        locked_iter->second.user_lock = true;
    } else {
        locked_info info = {false, true, 100};  // This number is not relevant

        current.locked_map[const_cast<void *>(ptr)] = info;
    }
}

void DefaultMemoryManager::userUnlock(const void *ptr) {
    this->unlock(const_cast<void *>(ptr), true);
}

bool DefaultMemoryManager::isUserLocked(const void *ptr) {
    memory_info &current = this->getCurrentMemoryInfo();
    lock_guard_t lock(this->memory_mutex);
    auto locked_iter = current.locked_map.find(const_cast<void *>(ptr));
    if (locked_iter == current.locked_map.end()) { return false; }
    return locked_iter->second.user_lock;
}

size_t DefaultMemoryManager::getMemStepSize() {
    lock_guard_t lock(this->memory_mutex);
    return this->mem_step_size;
}

void DefaultMemoryManager::setMemStepSize(size_t new_step_size) {
    lock_guard_t lock(this->memory_mutex);
    this->mem_step_size = new_step_size;
}

}  // namespace common
}  // namespace arrayfire
