/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

#include <memory>
#include <utility>

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: should this be 37?
#if AF_API_VERSION >= 36

    /**
     * An interface for defining functions used for memory management inside
     * ArrayFire. As is the case with the C++ API, while all functions are called,
     * not all functions need to be implemented (several are kept for legacy API
     * reasons). In these cases, returning a falsy value/having a NOOP is
     * acceptable, but all functions in the interface MUST be defined.
     *
     * ArrayFire defines native, backend and device-specific allocation
     * functions that rely on backend-specific specific ArrayFire APIs. These
     * should be called directly from functions defined on the af_memory_manager
     * implementation using the passed backend_manager pointer:
     * \code
       void *af_memory_manager_alloc(af_memory_manager* inst,
                                     const size_t size,
                                     bool user_lock)
       // ...
       // this function pointer will point to the internal, backend-defined
       // nativeAlloc function after af_set_memory_manager returns.
       backend_manager->nativeAlloc(...);
       // ...
       }
     * \endcode
     */
    typedef struct af_memory_manager {
      af_memory_manager* wrapper_handle;

      int (*af_memory_manager_get_active_device_id)(af_memory_manager* impl_);

      int (*af_memory_manager_get_max_memory_size)(
        af_memory_manager* impl_,
        int id
      );

      void* (*af_memory_manager_native_alloc)(af_memory_manager* impl_,
                                              size_t size);

      void (*af_memory_manager_native_free)(af_memory_manager* impl_,
                                                    void* ptr);
      
      void (*af_memory_manager_initialize)(af_memory_manager* inst);

      void (*af_memory_manager_shutdown)(af_memory_manager* inst);

      void (*af_memory_manager_set_max_memory_size)(af_memory_manager* inst);

      void (*af_memory_manager_add_memory_management)(
        af_memory_manager* inst,        
        int device
      );

      void (*af_memory_manager_remove_memory_management)(
        af_memory_manager* inst,        
        int device
      );

      void* (*af_memory_manager_alloc)(
        af_memory_manager* inst,
        const size_t size,
        bool user_lock
      );

      size_t (*af_memory_manager_allocated)(
        af_memory_manager* inst,
        void* ptr
      );

      void (*af_memory_manager_unlock)(
        af_memory_manager* inst,
        void* ptr,
        bool user_unlock
      );

      void (*af_memory_manager_buffer_info)(
        af_memory_manager* inst,
        size_t *alloc_bytes,
        size_t *alloc_buffers,
        size_t *lock_bytes,
        size_t *lock_buffers
      );

      void (*af_memory_manager_user_lock)(af_memory_manager* inst,
                                          const void* ptr);

      void (*af_memory_manager_user_unlock)(af_memory_manager* inst,
                                            const void* ptr);

      bool (*af_memory_manager_is_user_locked)(af_memory_manager* inst,
                                               const void* ptr);

      bool (*af_memory_manager_check_memory_limit)(af_memory_manager* inst);

      size_t (*af_memory_manager_get_max_bytes)(af_memory_manager* inst);

      unsigned (*af_memory_manager_get_max_buffers)(af_memory_manager* inst);

      void (*af_memory_manager_print_info)(
        af_memory_manager* inst,
        const char* msg,
        const int device
      );

      void (*af_memory_manager_garbage_collect)(af_memory_manager* inst);

      size_t (*af_memory_manager_get_mem_step_size)(af_memory_manager* inst);

      void (*af_memory_manager_set_mem_step_size)(af_memory_manager* inst,
                                                  size_t new_step_size
      );

    } af_memory_manager;

    /// \brief Sets the device memory manager for the current backend via
    /// the C api.
    AFAPI af_err af_set_memory_manager(af_memory_manager* manager,
                                       af_memory_manager_api_type api);

    /// \brief Sets the pinned device memory manager for the current backend
    AFAPI af_err af_set_pinned_memory_manager(af_memory_manager* manager,
                                              af_memory_manager_api_type api);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace spdlog {
class logger;
}

namespace af
{

// FIXME: should this be 37?
#if AF_API_VERSION >= 36

    /**
     * A generic interface that backend-specific memory managers implement.
     * NB: Overriding or defining methods on a class derived
     * from BackendMemoryManagerderived class will have NO EFFECT, since derived
     * internal ArrayFire implementations of backend memory managers are used
     * when a custom memory manager is specified.
     *
     * This declaration provides methods for a derived implementation of
     * MemoryManagerBase to call, which will be routed to the correct internal
     * ArrayFire implementation of some native device/backend-specific memory
     * operation.
    */
    class BackendMemoryManager {
      protected:
        std::shared_ptr<spdlog::logger> logger;
      public:
        BackendMemoryManager() = default;
        virtual ~BackendMemoryManager() = default;
        virtual int getActiveDeviceId() = 0;
        virtual size_t getMaxMemorySize(int id) = 0;
        virtual void* nativeAlloc(const size_t bytes) = 0;
        virtual void nativeFree(void *ptr) = 0;
        virtual spdlog::logger* getLogger() final {
            return this->logger.get();
        }
    };

    /**
     * An interface for writing custom memory managers.
     *
     * Functions in this interface are called by the rest of the ArrayFire API
     * during operations requiring memory (e.g. allocating Arrays, optimizing
     * allocations for the JIT, etc).
     */
    class MemoryManagerBase : public af_memory_manager
    {
     public:
      MemoryManagerBase() = default;
      virtual ~MemoryManagerBase() = default;

      // Performs any necessary initialization for the memory manager. Called
      // after the constructor returns and after the memory manager's
      // BackendMemoryManager is initialized (i.e. its methods can be called)
      virtual void initialize() = 0;

      // Called by the BackendMemoryManager dtor for each device; performs
      // any device or backend-specific cleanup. Since the BackendMemoryManager
      // is destroyed before any derived instance of MemoryManagerBase, this
      // is called before ~MemoryManagerBase is dispatched/invoked
      virtual void shutdown() = 0;

      virtual void setMaxMemorySize() = 0;

      // Intended to be used with OpenCL backend, where
      // users are allowed to add external devices(context, device pair)
      // to the list of devices automatically detected by the library
      virtual void addMemoryManagement(int device) = 0;

      // Intended to be used with OpenCL backend, where
      // users are allowed to add external devices(context, device pair)
      // to the list of devices automatically detected by the library
      virtual void removeMemoryManagement(int device) = 0;

      /// Returns a pointer of size at least long
      ///
      /// This funciton will return a memory location of at least \p size
      /// bytes. If there is already a free buffer available, it will use
      /// that buffer. Otherwise, it will allocate a new buffer using the
      /// nativeAlloc function.
      virtual void *alloc(const size_t size, bool user_lock) = 0;

      /// returns the size of the buffer at the pointer allocated by the memory
      /// manager.
      virtual size_t allocated(void *ptr) = 0;

      /// Frees or marks the pointer for deletion during the nex garbage collection
      /// event
      virtual void unlock(void *ptr, bool user_unlock) = 0;

      virtual void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                              size_t *lock_bytes,  size_t *lock_buffers) = 0;
      virtual void userLock(const void *ptr) = 0;
      virtual void userUnlock(const void *ptr) = 0;
      virtual bool isUserLocked(const void *ptr) = 0;

      virtual bool checkMemoryLimit() = 0;

      virtual size_t getMaxBytes() = 0;
      virtual unsigned getMaxBuffers() = 0;

      virtual void printInfo(const char *msg, const int device) = 0;

      /// Optional legacy methods based on the default implementation and API. These are
      /// user-defined, such that they will never be called by any backend's memory manager.
      /// Throwing an exception is reasonable behavior if not implemented.
      virtual void garbageCollect() = 0;
      virtual size_t getMemStepSize() = 0;
      virtual void setMemStepSize(size_t new_step_size) = 0;

      /// Sets a BackendMemoryManager for this memory manager implementation.
      virtual void setBackendManager(std::unique_ptr<BackendMemoryManager> manager) final {
          backendMem_ = std::move(manager);
      }

    protected:
      /// A backend-specific collection of methods for device-specific
      /// memory allocation facilities including native allocation and frees,
      /// device ID acquisition, and max-memory lookup. Callable by derived
      /// implementations of MemoryManagerBase.
      /// backendMem_ will be created and set on the derived implementation
      /// when `af::setMemoryManager` returns such that calls to these
      /// methods are correctly dispatched to the proper internal ArrayFire
      /// implementation.
      std::unique_ptr<BackendMemoryManager> backendMem_;
      
    public:
      virtual int getActiveDeviceId() final {
          return backendMem_->getActiveDeviceId();
      }

      virtual size_t getMaxMemorySize(int id) final {
          return backendMem_->getMaxMemorySize(id);
      }

      virtual void* nativeAlloc(const size_t bytes) final {
          return backendMem_->nativeAlloc(bytes);
      }

      virtual void nativeFree(void *ptr) final {
          backendMem_->nativeFree(ptr);
      }

      virtual spdlog::logger* getLogger() final {
          return backendMem_->getLogger();
      }
    };

    /// \brief Sets the device memory manager for the current backend
    AFAPI void setMemoryManager(af_memory_manager* manager);

    AFAPI void setPinnedMemoryManager(af_memory_manager* manager);

#endif

}
#endif
