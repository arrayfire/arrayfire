/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "ArrayFireDefaultMemoryManager.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(BSD) && !defined(__gnu_hurd__)
#include <sys/sysctl.h>
#endif

#else
#define NOMEMORYSIZE
#endif

#include <cstdlib>

#define divup(a, b) (((a) + (b)-1) / (b))

using std::lock_guard;
using std::max;
using std::move;
using std::mutex;
using std::stoi;
using std::string;
using std::vector;

/**
 * Returns the size of physical memory (RAM) in bytes.
 */
size_t getHostMemorySize() {
#if defined(_WIN32) && (defined(__CYGWIN__) || defined(__CYGWIN32__))
    /* Cygwin under Windows. ------------------------------------ */
    /* New 64-bit MEMORYSTATUSEX isn't available.  Use old 32.bit */
    MEMORYSTATUS status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatus(&status);
    return (size_t)status.dwTotalPhys;

#elif defined(_WIN32)
    /* Windows. ------------------------------------------------- */
    /* Use new 64-bit MEMORYSTATUSEX, not old 32-bit MEMORYSTATUS */
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;

#elif defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
    /* UNIX variants. ------------------------------------------- */
    /* Prefer sysctl() over sysconf() except sysctl() HW_REALMEM and HW_PHYSMEM
     */

#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
    int mib[2];
    mib[0]       = CTL_HW;
#if defined(HW_MEMSIZE)
    mib[1]       = HW_MEMSIZE; /* OSX. --------------------- */
#elif defined(HW_PHYSMEM64)
    mib[1] = HW_PHYSMEM64; /* NetBSD, OpenBSD. --------- */
#endif
    int64_t size = 0;          /* 64-bit */
    size_t len   = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) return (size_t)size;
    return 0L; /* Failed? */

#elif defined(_SC_AIX_REALMEM)
    /* AIX. ----------------------------------------------------- */
    return (size_t)sysconf(_SC_AIX_REALMEM) * (size_t)1024L;

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    /* FreeBSD, Linux, OpenBSD, and Solaris. -------------------- */
    return static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
           static_cast<size_t>(sysconf(_SC_PAGESIZE));

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    /* Legacy. -------------------------------------------------- */
    return (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);

#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM))
    /* DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX. -------- */
    int mib[2];
    mib[0]            = CTL_HW;
#if defined(HW_REALMEM)
    mib[1]            = HW_REALMEM; /* FreeBSD. ----------------- */
#elif defined(HW_PYSMEM)
    mib[1] = HW_PHYSMEM; /* Others. ------------------ */
#endif
    unsigned int size = 0;          /* 32-bit */
    size_t len        = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) return (size_t)size;
    return 0L; /* Failed? */
#endif /* sysctl and sysconf variants */

#else
    return 0L; /* Unknown OS. */
#endif
    return 0;
}

ArrayFireDefaultMemoryManager::memory_info &
ArrayFireDefaultMemoryManager::getCurrentMemoryInfo() {
    return memory;
}

void ArrayFireDefaultMemoryManager::cleanDeviceMemoryManager() {
    if (this->debug_mode) { return; }

    // This vector is used to store the pointers which will be deleted by
    // the memory manager. We are using this to avoid calling free while
    // the lock is being held because the CPU backend calls sync.
    vector<void *> free_ptrs;
    size_t bytes_freed                                  = 0;
    ArrayFireDefaultMemoryManager::memory_info &current = memory;
    {
        lock_guard<mutex> lock(this->memory_mutex);
        // Return if all buffers are locked
        if (current.total_buffers == current.lock_buffers) { return; }
        free_ptrs.reserve(current.free_map.size());

        for (auto &kv : current.free_map) {
            size_t num_ptrs = kv.second.size();
            // Free memory by pushing the last element into the free_ptrs
            // vector which will be freed once outside of the lock
            // for (auto ptr : kv.second) { free_ptrs.emplace_back(pair); }
            move(begin(kv.second), end(kv.second), back_inserter(free_ptrs));
            current.total_bytes -= num_ptrs * kv.first;
            bytes_freed += num_ptrs * kv.first;
            current.total_buffers -= num_ptrs;
        }
        current.free_map.clear();
    }

    //  Free memory outside of the lock
    for (auto *ptr : free_ptrs) {
        af_memory_manager_native_free(0, ptr);
    }
}

ArrayFireDefaultMemoryManager::ArrayFireDefaultMemoryManager(
    unsigned max_buffers, bool debug)
    : mem_step_size(1 << 20)
    , max_buffers(max_buffers)
    , debug_mode(debug)
    , memory() {}

void ArrayFireDefaultMemoryManager::initialize() { this->setMaxMemorySize(); }

void ArrayFireDefaultMemoryManager::shutdown() { signalMemoryCleanup(); }

size_t ArrayFireDefaultMemoryManager::getMaxMemorySize() {
    return getHostMemorySize();
}

void ArrayFireDefaultMemoryManager::setMaxMemorySize() {
    size_t memsize = this->getMaxMemorySize();
    memory.max_bytes =
        memsize == 0
            ? ONE_GB
            : max(memsize * 0.75, static_cast<double>(memsize - ONE_GB));
}

float ArrayFireDefaultMemoryManager::getMemoryPressure() {
    lock_guard<mutex> lock(this->memory_mutex);
    memory_info &current = this->getCurrentMemoryInfo();
    if (current.lock_bytes > current.max_bytes ||
        current.lock_buffers > max_buffers) {
        return 1.0;
    } else {
        return 0.0;
    }
}

bool ArrayFireDefaultMemoryManager::jitTreeExceedsMemoryPressure(size_t bytes) {
    lock_guard<mutex> lock(this->memory_mutex);
    memory_info &current = this->getCurrentMemoryInfo();
    return 2 * bytes > current.lock_bytes;
}

void *ArrayFireDefaultMemoryManager::alloc(bool user_lock, const unsigned ndims,
                                           long long *dims,
                                           const unsigned element_size) {
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
                // printf(
                //     "current.lock_bytes(%zu) >= current.max_bytes(%zu) || "
                //     "current.total_buffers(%zu) >= this->max_buffers(%u)\n",
                //     current.lock_bytes, current.max_bytes,
                //     current.total_buffers, this->max_buffers);

                this->signalMemoryCleanup();
            }

            lock_guard<mutex> lock(this->memory_mutex);
            auto free_buffer_iter = current.free_map.find(alloc_bytes);
            if (free_buffer_iter != current.free_map.end() &&
                !free_buffer_iter->second.empty()) {
                // Delete existing buffer info and underlying event
                // Set to existing in from free map
                vector<void *> &free_buffer_vector = free_buffer_iter->second;
                ptr                                = free_buffer_vector.back();
                // printf("reusing: %p @ %zu\n", ptr, alloc_bytes);
                free_buffer_vector.pop_back();
                current.locked_map[ptr] = info;
                current.lock_bytes += alloc_bytes;
                current.lock_buffers++;
            }
        }

        // Only comes here if buffer size not found or in debug mode
        if (ptr == nullptr) {
            // Perform garbage collection if memory can not be allocated
          af_memory_manager_native_alloc(0, &ptr, alloc_bytes);
            if (!ptr) {
                // If out of memory, run garbage collect and try again
                // if (ex.getError() != AF_ERR_NO_MEM) { throw; }
                this->signalMemoryCleanup();
                af_memory_manager_native_alloc(0, &ptr, alloc_bytes);
            }
            lock_guard<mutex> lock(this->memory_mutex);
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

size_t ArrayFireDefaultMemoryManager::allocated(void *ptr) {
    if (!ptr) { return 0; }
    memory_info &current = this->getCurrentMemoryInfo();
    auto locked_iter     = current.locked_map.find(ptr);
    if (locked_iter == current.locked_map.end()) { return 0; }
    return (locked_iter->second).bytes;
}

void ArrayFireDefaultMemoryManager::unlock(void *ptr, bool user_unlock) {
    // Shortcut for empty arrays
    if (!ptr) { return; }

    // Frees the pointer outside the lock.
    uptr_t freed_ptr(nullptr, [this](void *p) {
        af_memory_manager_native_free(0, p);
    });
    {
        lock_guard<mutex> lock(this->memory_mutex);
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

void ArrayFireDefaultMemoryManager::signalMemoryCleanup() {
    cleanDeviceMemoryManager();
}

void ArrayFireDefaultMemoryManager::printInfo(const char *msg,
                                              const int device) {
    const memory_info &current = this->getCurrentMemoryInfo();

    printf("%s\n", msg);
    printf(
        "---------------------------------------------------------\n"
        "|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |\n"
        "---------------------------------------------------------\n");

    lock_guard<mutex> lock(this->memory_mutex);
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

void ArrayFireDefaultMemoryManager::usageInfo(size_t *alloc_bytes,
                                              size_t *alloc_buffers,
                                              size_t *lock_bytes,
                                              size_t *lock_buffers) {
    const memory_info &current = this->getCurrentMemoryInfo();
    lock_guard<mutex> lock(this->memory_mutex);
    if (alloc_bytes) { *alloc_bytes = current.total_bytes; }
    if (alloc_buffers) { *alloc_buffers = current.total_buffers; }
    if (lock_bytes) { *lock_bytes = current.lock_bytes; }
    if (lock_buffers) { *lock_buffers = current.lock_buffers; }
}

void ArrayFireDefaultMemoryManager::userLock(const void *ptr) {
    memory_info &current = this->getCurrentMemoryInfo();

    lock_guard<mutex> lock(this->memory_mutex);

    auto locked_iter = current.locked_map.find(const_cast<void *>(ptr));
    if (locked_iter != current.locked_map.end()) {
        locked_iter->second.user_lock = true;
    } else {
        locked_info info = {false, true, 100};  // This number is not relevant

        current.locked_map[const_cast<void *>(ptr)] = info;
    }
}

void ArrayFireDefaultMemoryManager::userUnlock(const void *ptr) {
    this->unlock(const_cast<void *>(ptr), true);
}

bool ArrayFireDefaultMemoryManager::isUserLocked(const void *ptr) {
    memory_info &current = this->getCurrentMemoryInfo();
    lock_guard<mutex> lock(this->memory_mutex);
    auto locked_iter = current.locked_map.find(const_cast<void *>(ptr));
    if (locked_iter == current.locked_map.end()) { return false; }
    return locked_iter->second.user_lock;
}

size_t ArrayFireDefaultMemoryManager::getMemStepSize() {
    lock_guard<mutex> lock(this->memory_mutex);
    return this->mem_step_size;
}

void ArrayFireDefaultMemoryManager::setMemStepSize(size_t new_step_size) {
    lock_guard<mutex> lock(this->memory_mutex);
    this->mem_step_size = new_step_size;
}
