/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Event.hpp>
#include <backend.hpp>
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

constexpr unsigned MAX_BUFFERS = 1000;
constexpr size_t ONE_GB        = 1 << 30;

struct MemoryEventPair {
    void *ptr;
    detail::Event e;
    MemoryEventPair(MemoryEventPair &other)  = delete;
    MemoryEventPair(MemoryEventPair &&other) = default;
    MemoryEventPair &operator=(MemoryEventPair &&other) = default;
    MemoryEventPair &operator=(MemoryEventPair &other) = delete;
};

template<typename T>
class MemoryManager {
    struct locked_info {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    };

    using locked_t    = typename std::unordered_map<void *, locked_info>;
    using locked_iter = typename locked_t::iterator;

    using free_t    = std::unordered_map<size_t, std::vector<MemoryEventPair>>;
    using free_iter = typename free_t::iterator;

    using uptr_t = std::unique_ptr<void, std::function<void(void *)>>;

    struct memory_info {
        locked_t locked_map;
        free_t free_map;

        size_t lock_bytes;
        size_t lock_buffers;
        size_t total_bytes;
        size_t total_buffers;
        size_t max_bytes;

        memory_info()
            // Calling getMaxMemorySize() here calls the virtual function
            // that returns 0 Call it from outside the constructor.
            : max_bytes(ONE_GB)
            , total_bytes(0)
            , total_buffers(0)
            , lock_bytes(0)
            , lock_buffers(0) {}

        memory_info(memory_info &other)  = delete;
        memory_info(memory_info &&other) = default;
        memory_info &operator=(memory_info &other) = delete;
        memory_info &operator=(memory_info &&other) = default;
    };

    size_t mem_step_size;
    unsigned max_buffers;
    std::vector<memory_info> memory;
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
    MemoryEventPair alloc(const size_t size, bool user_lock);

    /// returns the size of the buffer at the pointer allocated by the memory
    /// manager.
    size_t allocated(void *ptr);

    /// Frees or marks the pointer for deletion during the nex garbage
    /// collection event
    void unlock(void *ptr, detail::Event &&e, bool user_unlock);

    /// Frees all buffers which are not locked by the user or not being
    /// used.
    void garbageCollect();

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
    inline void *nativeAlloc(const size_t bytes);
    inline void nativeFree(void *ptr);
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
};

}  // namespace common
