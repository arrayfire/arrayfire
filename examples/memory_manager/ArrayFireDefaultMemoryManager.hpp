/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/memory.h>

#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstddef>
#include <mutex>

constexpr unsigned MAX_BUFFERS = 1000;
constexpr size_t ONE_GB        = 1 << 30;


using uptr_t = std::unique_ptr<void, std::function<void(void *)>>;

class ArrayFireDefaultMemoryManager {
    size_t mem_step_size;
    unsigned max_buffers;

    bool debug_mode;

    struct locked_info {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    };

    using locked_t = typename std::unordered_map<void *, locked_info>;
    using free_t   = std::unordered_map<size_t, std::vector<void *>>;

    struct memory_info {
        locked_t locked_map;
        free_t free_map;

        size_t max_bytes;
        size_t total_bytes;
        size_t total_buffers;
        size_t lock_bytes;
        size_t lock_buffers;

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

    memory_info &getCurrentMemoryInfo();

   public:
    ArrayFireDefaultMemoryManager(unsigned max_buffers, bool debug);

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void addMemoryManagement(int device);

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void removeMemoryManagement(int device);

    void setMaxMemorySize();

    void initialize();
    void shutdown();
    size_t getMaxMemorySize();

    size_t getDeviceMemorySize(int device);


    /// Returns a pointer of size at least long
    ///
    /// This funciton will return a memory location of at least \p size
    /// bytes. If there is already a free buffer available, it will use
    /// that buffer. Otherwise, it will allocate a new buffer using the
    /// nativeAlloc function.
    void *alloc(bool user_lock, const unsigned ndims, long long *dims,
                const unsigned element_size);

    /// returns the size of the buffer at the pointer allocated by the memory
    /// manager.
    size_t allocated(void *ptr);

    /// Frees or marks the pointer for deletion during the nex garbage
    /// collection event
    void unlock(void *ptr, bool user_unlock);

    /// Frees all buffers which are not locked by the user or not being
    /// used.
    void signalMemoryCleanup();

    void printInfo(const char *msg, const int device);
    void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                   size_t *lock_bytes, size_t *lock_buffers);
    void userLock(const void *ptr);
    void userUnlock(const void *ptr);
    bool isUserLocked(const void *ptr);
    size_t getMemStepSize();
    void setMemStepSize(size_t new_step_size);
    float getMemoryPressure();
    bool jitTreeExceedsMemoryPressure(size_t bytes);

    ~ArrayFireDefaultMemoryManager() = default;

   protected:
    ArrayFireDefaultMemoryManager()                                  = delete;
    ArrayFireDefaultMemoryManager(const ArrayFireDefaultMemoryManager &other) = delete;
    ArrayFireDefaultMemoryManager(ArrayFireDefaultMemoryManager &&other)      = default;
    ArrayFireDefaultMemoryManager &operator=(const ArrayFireDefaultMemoryManager &other) = delete;
    ArrayFireDefaultMemoryManager &operator=(ArrayFireDefaultMemoryManager &&other) = default;
    std::mutex memory_mutex;

    af_memory_manager native_allocator_access_;
    // backend-specific
    memory_info memory;
    // backend-agnostic
    void cleanDeviceMemoryManager();
  };

