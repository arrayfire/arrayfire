/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Event.hpp>
#include <common/AllocatorInterface.hpp>

#include <cstddef>
#include <memory>

namespace spdlog {
class logger;
}

namespace arrayfire {
namespace common {
/**
 * A internal base interface for a memory manager which is exposed to AF
 * internals. Externally, both the default AF memory manager implementation and
 * custom memory manager implementations are wrapped in a derived implementation
 * of this interface.
 */
class MemoryManagerBase {
   public:
    MemoryManagerBase()                                     = default;
    MemoryManagerBase &operator=(const MemoryManagerBase &) = delete;
    MemoryManagerBase(const MemoryManagerBase &)            = delete;
    virtual ~MemoryManagerBase() {}
    // Shuts down the allocator interface which calls shutdown on the subclassed
    // memory manager with device-specific context
    virtual void shutdownAllocator() {
        if (nmi_) nmi_->shutdown();
    }
    virtual void initialize()                                        = 0;
    virtual void shutdown()                                          = 0;
    virtual void *alloc(bool user_lock, const unsigned ndims, dim_t *dims,
                        const unsigned element_size)                 = 0;
    virtual size_t allocated(void *ptr)                              = 0;
    virtual void unlock(void *ptr, bool user_unlock)                 = 0;
    virtual void signalMemoryCleanup()                               = 0;
    virtual void printInfo(const char *msg, const int device)        = 0;
    virtual void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                           size_t *lock_bytes, size_t *lock_buffers) = 0;
    virtual void userLock(const void *ptr)                           = 0;
    virtual void userUnlock(const void *ptr)                         = 0;
    virtual bool isUserLocked(const void *ptr)                       = 0;
    virtual size_t getMemStepSize()                                  = 0;
    virtual void setMemStepSize(size_t new_step_size)                = 0;

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

    // Memory pressure functions
    void setMemoryPressureThreshold(float pressure) {
        memoryPressureThreshold_ = pressure;
    }
    float getMemoryPressureThreshold() const {
        return memoryPressureThreshold_;
    }
    virtual float getMemoryPressure()                       = 0;
    virtual bool jitTreeExceedsMemoryPressure(size_t bytes) = 0;

   private:
    // A threshold at or above which JIT evaluations will be triggered due to
    // memory pressure. Settable via a call to setMemoryPressureThreshold
    float memoryPressureThreshold_{1.0};
    // A backend-specific memory manager, containing backend-specific
    // methods that call native memory manipulation functions in a device
    // API. We need to wrap these since they are opaquely called by the
    // memory manager.
    std::unique_ptr<AllocatorInterface> nmi_;
};

}  // namespace common
}  // namespace arrayfire
