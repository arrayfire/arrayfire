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
#include <events.hpp>
#include <af/memory.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef AF_MEM_DEBUG
#define AF_MEM_DEBUG 0
#endif

namespace spdlog {
class logger;
}
namespace common {
using mutex_t      = std::mutex;
using lock_guard_t = std::lock_guard<mutex_t>;

constexpr unsigned MAX_BUFFERS = 1000;
constexpr size_t ONE_GB        = 1 << 30;

namespace memory {

/**
 * An interface that provides backend-specific memory management functions,
 * typically calling a dedicated backend-specific native API. Stored, wrapped,
 * and called by a MemoryManagerBase, from which calls to its interface are
 * delegated.
 */
class AllocatorInterface {
   public:
    AllocatorInterface()                          = default;
    virtual ~AllocatorInterface()                 = default;
    virtual int getActiveDeviceId()               = 0;
    virtual size_t getMaxMemorySize(int id)       = 0;
    virtual void *nativeAlloc(const size_t bytes) = 0;
    virtual void nativeFree(void *ptr)            = 0;
    virtual spdlog::logger *getLogger() final { return this->logger.get(); }

   protected:
    std::shared_ptr<spdlog::logger> logger;
};

/**
 * A internal base interface for a memory manager which is exposed to AF
 * internals. Externally, both the default AF memory manager implementation and
 * custom memory manager implementations are wrapped in a derived implementation
 * of this interface.
 */
class MemoryManagerBase {
   public:
    virtual void initialize()                                        = 0;
    virtual void shutdown()                                          = 0;
    virtual af_buffer_info alloc(const size_t size, bool user_lock)  = 0;
    virtual size_t allocated(void *ptr)                              = 0;
    virtual void unlock(void *ptr, af_event e, bool user_unlock)     = 0;
    virtual void garbageCollect()                                    = 0;
    virtual void printInfo(const char *msg, const int device)        = 0;
    virtual void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                           size_t *lock_bytes, size_t *lock_buffers) = 0;
    virtual void userLock(const void *ptr)                           = 0;
    virtual void userUnlock(const void *ptr)                         = 0;
    virtual bool isUserLocked(const void *ptr)                       = 0;
    virtual size_t getMemStepSize()                                  = 0;
    virtual size_t getMaxBytes()                                     = 0;
    virtual unsigned getMaxBuffers()                                 = 0;
    virtual void setMemStepSize(size_t new_step_size)                = 0;
    virtual bool checkMemoryLimit()                                  = 0;

    /// Backend-specific functions
    // OpenCL
    virtual void addMemoryManagement(int device)    = 0;
    virtual void removeMemoryManagement(int device) = 0;

    int getActiveDeviceId() { return nmi_->getActiveDeviceId(); }
    size_t getMaxMemorySize(int id) { return nmi_->getMaxMemorySize(id); }
    void *nativeAlloc(const size_t bytes) { return nmi_->nativeAlloc(bytes); }
    void nativeFree(void *ptr) { nmi_->nativeFree(ptr); }
    virtual spdlog::logger *getLogger() final { return nmi_->getLogger(); }
    virtual void setAllocator(std::unique_ptr<AllocatorInterface> nmi) {
        nmi_ = std::move(nmi);
    }

   private:
    // A backend-specific memory manager, containing backend-specific
    // methods that call native memory manipulation functions in a device
    // API. We need to wrap these since they are opaquely called by the
    // memory manager.
    std::unique_ptr<AllocatorInterface> nmi_;
};

/******************** Default memory manager implementation *******************/

struct locked_info {
    bool manager_lock;
    bool user_lock;
    size_t bytes;
};

using locked_t    = typename std::unordered_map<void *, memory::locked_info>;
using locked_iter = typename locked_t::iterator;

using free_t    = std::unordered_map<size_t, std::vector<af_buffer_info>>;
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

}  // namespace memory

class DefaultMemoryManager : public memory::MemoryManagerBase {
    size_t mem_step_size;
    unsigned max_buffers;

    bool debug_mode;

    memory::memory_info &getCurrentMemoryInfo();

   public:
    DefaultMemoryManager(int num_devices, unsigned max_buffers, bool debug);

    // Initializes the memory manager
    virtual void initialize() override;

    // Shuts down the memory manager
    virtual void shutdown() override;

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void addMemoryManagement(int device) override;

    // Intended to be used with OpenCL backend, where
    // users are allowed to add external devices(context, device pair)
    // to the list of devices automatically detected by the library
    void removeMemoryManagement(int device) override;

    void setMaxMemorySize();

    /// Returns a pointer of size at least long
    ///
    /// This funciton will return a memory location of at least \p size
    /// bytes. If there is already a free buffer available, it will use
    /// that buffer. Otherwise, it will allocate a new buffer using the
    /// nativeAlloc function.
    af_buffer_info alloc(const size_t size, bool user_lock) override;

    /// returns the size of the buffer at the pointer allocated by the memory
    /// manager.
    size_t allocated(void *ptr) override;

    /// Frees or marks the pointer for deletion during the nex garbage
    /// collection event
    void unlock(void *ptr, af_event e, bool user_unlock) override;

    /// Frees all buffers which are not locked by the user or not being
    /// used.
    void garbageCollect() override;

    void printInfo(const char *msg, const int device) override;
    void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                   size_t *lock_bytes, size_t *lock_buffers) override;
    void userLock(const void *ptr) override;
    void userUnlock(const void *ptr) override;
    bool isUserLocked(const void *ptr) override;
    size_t getMemStepSize() override;
    size_t getMaxBytes() override;
    unsigned getMaxBuffers() override;
    void setMemStepSize(size_t new_step_size) override;
    bool checkMemoryLimit() override;

   protected:
    DefaultMemoryManager()                                   = delete;
    ~DefaultMemoryManager()                                  = default;
    DefaultMemoryManager(const DefaultMemoryManager &other)  = delete;
    DefaultMemoryManager(const DefaultMemoryManager &&other) = delete;
    DefaultMemoryManager &operator=(const DefaultMemoryManager &other) = delete;
    DefaultMemoryManager &operator=(const DefaultMemoryManager &&other) =
        delete;
    mutex_t memory_mutex;
    // backend-specific
    std::vector<common::memory::memory_info> memory;
    // backend-agnostic
    void cleanDeviceMemoryManager(int device);
};

}  // namespace common
