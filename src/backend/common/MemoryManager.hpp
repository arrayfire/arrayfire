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

namespace spdlog {
class logger;
}
namespace common {
using mutex_t      = std::mutex;
using lock_guard_t = std::lock_guard<mutex_t>;

const unsigned MAX_BUFFERS = 1000;
const size_t ONE_GB        = 1 << 30;

template<typename T>
class MemoryManager {
    typedef struct {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    } locked_info;

using locked_t    = typename std::unordered_map<void *, locked_info>;
using free_t    = std::unordered_map<size_t, std::vector<void *> >;

    using free_t    = std::unordered_map<size_t, std::vector<void *>>;
    using free_iter = free_t::iterator;

    using uptr_t = std::unique_ptr<void, std::function<void(void *)>>;

    typedef struct memory_info {
        locked_t locked_map;
        free_t free_map;

        size_t lock_bytes;
        size_t lock_buffers;
        size_t total_bytes;
        size_t total_buffers;
        size_t max_bytes;

        memory_info() {
            // Calling getMaxMemorySize() here calls the virtual function that
            // returns 0 Call it from outside the constructor.
            max_bytes     = ONE_GB;
            total_bytes   = 0;
            total_buffers = 0;
            lock_bytes    = 0;
            lock_buffers  = 0;
        }
    } memory_info;

    size_t mem_step_size;
    unsigned max_buffers;
    
    std::shared_ptr<spdlog::logger> logger;
    bool debug_mode;

    memory_info &getCurrentMemoryInfo();

    inline int getActiveDeviceId();
    inline size_t getMaxMemorySize(int id);
    void cleanDeviceMemoryManager(int device);

   public:
    MemoryManager(int num_devices, unsigned max_buffers, bool debug);

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void addMemoryManagement(int device);

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void removeMemoryManagement(int device);

    void setMaxMemorySize();

    /// Returns a pointer of size at least long
    ///
    /// This funciton will return a memory location of at least \p size
    /// bytes. If there is already a free buffer available, it will use
    /// that buffer. Otherwise, it will allocate a new buffer using the
    /// nativeAlloc function.
    void *alloc(const size_t size, bool user_lock);

    /// returns the size of the buffer at the pointer allocated by the memory
    /// manager.
    size_t allocated(void *ptr);

    /// Frees or marks the pointer for deletion during the nex garbage
    /// collection event
    void unlock(void *ptr, bool user_unlock);

    /// Frees all buffers which are not locked by the user or not being used.
    virtual void garbageCollect() = 0;

    void printInfo(const char *msg, const int device);
    void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                    size_t *lock_bytes, size_t *lock_buffers);
    void userLock(const void *ptr);
    void userUnlock(const void *ptr);
    bool isUserLocked(const void *ptr);
    size_t getMemStepSize();
    size_t getMaxBytes();
    unsigned getMaxBuffers();
    void setMemStepSize(size_t new_step_size);
    virtual void *nativeAlloc(const size_t bytes) = 0;
    virtual void nativeFree(void *ptr) = 0;
    bool checkMemoryLimit();

   protected:
    spdlog::logger *getLogger();
    MemoryManager()                            = delete;
    ~MemoryManager()                           = default;
    MemoryManager(const MemoryManager &other)  = delete;
    MemoryManager(const MemoryManager &&other) = delete;
    MemoryManager &operator=(const MemoryManager &other) = delete;
    MemoryManager &operator=(const MemoryManager &&other) = delete;
    mutex_t memory_mutex;
    // backend-specific
    std::vector<common::memory::memory_info> memory;
    // backend-agnostic
    void cleanDeviceMemoryManager(int device);
};

}  // namespace common
