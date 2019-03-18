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
     * \brief A C interface for defining functions for a custom memory
     * manager to be used by ArrayFire.
     *
     * Functions in this interface are called by the rest of the ArrayFire API
     * during operations requiring memory (e.g. allocating Arrays, optimizing
     * allocations for the JIT, etc).
     *
     * As is the case with the C++ API, while all functions must be given some
     * function body, not all need to be explicitly implemented (several are
     * kept for legacy API reasons. Others are opinionated based on the 
     * specific memory manager implementation (e.g. garbage collection).
     * In these cases, returning a falsy value/having an empty function body
     * is acceptable, but all functions MUST be defined.
     *
     * ArrayFire defines native, backend and device-specific allocation
     * functions that rely on backend-specific ArrayFire APIs. These
     * should be called directly from functions defined on the af_memory_manager
     * implementation using the passed af_memory_manager pointer:
     *
     * \code
       void *af_memory_manager_alloc(af_memory_manager* inst,
                                     const size_t size,
                                     bool user_lock) {
         // ...
         // this function pointer will point to the internal, backend-defined
         // nativeAlloc function after af_set_memory_manager returns.
         void* ptr = inst->nativeAlloc(...);
         // ...
         // [use ptr in your custom implementation]
       }
       \endcode
     *
     * C-style struct inheritance can and should be used to extend the memory
     * manager and define custom methods and struct members on top of the
     * existing implementation.
     *
     * \ingroup memory_manager_c_api
     */
    typedef struct af_memory_manager {
      /**
       * \brief Performs any necessary post-initialization for
       * the memory manager.
       *
       * \note Called when \ref af_set_memory_manager or
       * \ref af_set_pinned_memory_manager is invoked. Backend-specific
       *  methods are callable (e.g. af_memory_manager_native_alloc)
       */
      void (*af_memory_manager_initialize)(af_memory_manager* inst);

      /**
       * \brief Called when shutting down the memory manager, before
       * the manager itself is freed. Called for each device, and can
       * perform device or backend-specific cleanup
       */
      void (*af_memory_manager_shutdown)(af_memory_manager* inst);

      /**
       * \brief Adds an external OpenCL device to the memory manager.
       *
       * \param[in] device device id of the new device
       *
       * \note Intended to be used with OpenCL backend, where
       * users are allowed to add external devices(context, device pair)
       * to the list of devices automatically detected by the library
       */
      void (*af_memory_manager_add_memory_management)(
        af_memory_manager* inst,        
        int device
      );

      /**
       * \brief Removes an external OpenCL device to the memory manager.
       *
       * \param[in] device device id of the new device
       *
       * \note Intended to be used with OpenCL backend, where
       * users are allowed to add external devices(context, device pair)
       * to the list of devices automatically detected by the library
       */
      void (*af_memory_manager_remove_memory_management)(
        af_memory_manager* inst,        
        int device
      );

      /**
       * \brief Returns a pointer of size at least `size`.
       *
       * \param[in] size the size of the buffer to allocate
       * \param[in] user_lock whether or not the memory was allocated via a
       * call to af::alloc/af_alloc_device or internally by ArrayFire
       */
      void* (*af_memory_manager_alloc)(
        af_memory_manager* inst,
        const size_t size,
        bool user_lock
      );

      /** \brief Returns the size of the buffer at the pointer allocated by
       * the memory manager.
       *
       * \param[in] ptr pointer to check allocation size
       */
      size_t (*af_memory_manager_allocated)(
        af_memory_manager* inst,
        void* ptr
      );

      /**
       * \brief Frees the pointer (or marks as unused).
       *
       * \note ArrayFire will call this function when memory is no longer needed.
       * Depending on implementation, this can either free memory immediately
       * or mark it for later deletion.
       *
       * \param[in] ptr the pointer to unlock.
       * \param[in] user_unlock whether or not the call to unlock is via the user
       * with af::free/af_free_device or internally by ArrayFire
       */
      void (*af_memory_manager_unlock)(
        af_memory_manager* inst,
        void* ptr,
        bool user_unlock
      );

      /**
       * \brief Gives information about memory manager state.
       *
       * \note This information is used by the ArrayFire JIT.
       *
       * \param[out] alloc_bytes the number of bytes allocated by the manager
       * \param[out] alloc_buffers the number of buffers created by the manager
       * \param[out] lock_bytes the number of bytes in use
       * \param[out] lock_buffers the number of buffers in use
       */
      void (*af_memory_manager_buffer_info)(
        af_memory_manager* inst,
        size_t *alloc_bytes,
        size_t *alloc_buffers,
        size_t *lock_bytes,
        size_t *lock_buffers
      );

      /**
       * \brief Locks memory in the manager as user-controlled
       *
       * \note Generally called when ArrayFire returns a pointer to the user,
       * as is the case when acquiring a device pointer
       *
       * \param[in] ptr the pointer to lock
       */
      void (*af_memory_manager_user_lock)(af_memory_manager* inst,
                                          const void* ptr);

      /**
       * \brief Releases memory back to the manager from the user
       *
       * \param[in] ptr the pointer to release
       */
      void (*af_memory_manager_user_unlock)(af_memory_manager* inst,
                                            const void* ptr);

      /**
       * \brief Returns if a piece of memory is locked by the user.
       *
       * \param[in] ptr memory to check
       */
      bool (*af_memory_manager_is_user_locked)(af_memory_manager* inst,
                                               const void* ptr);

      /**
       * \brief Returns whether allocated memory is at an
       * implementation-defined limit (generally per-device)
       *
       * \note This is an implementation-specific function used by the JIT
       */
      bool (*af_memory_manager_check_memory_limit)(af_memory_manager* inst);

      /**
       * \brief Returns the maximum number of bytes that can be allocated
       * by the manager (generally per-device)
       *
       * \note Used by the JIT
       */
      size_t (*af_memory_manager_get_max_bytes)(af_memory_manager* inst);

      /**
       * \brief Returns the maximum number of buffers that can be allocated
       * by the manager (generally per-device)
       *
       * \note Used by the JIT
       */
      unsigned (*af_memory_manager_get_max_buffers)(af_memory_manager* inst);

      /**
       * \brief Prints information about current memory manager state
       *
       * \param[in] msg a idenfier string with which the message can
       * be prefixed
       * \param[in] device the device to print status about
       */
      void (*af_memory_manager_print_info)(
        af_memory_manager* inst,
        const char* msg,
        const int device
      );

      /**
       * \brief Cleans up unused memory.
       *
       * \note While not all memory manager implementations will support
       * garbage collection, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      void (*af_memory_manager_garbage_collect)(af_memory_manager* inst);

      /**
       * \brief Gets the memory step size of the manager.
       *
       * \note While not all memory manager implementations will support
       * a step size detail, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      size_t (*af_memory_manager_get_mem_step_size)(af_memory_manager* inst);

      /**
       * \brief Sets the memory step size of the manager.
       *
       * \param[in] new_step_size the new step size for the memory manager.
       *
       * \note While not all memory manager implementations will support
       * a step size detail, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      void (*af_memory_manager_set_mem_step_size)(af_memory_manager* inst,
                                                  size_t new_step_size
      );

      /**
       * Backend-specific methods callable by a struct that inherits
       * af_memory_manager members.
       *
       * These should NOT BE OVERRIDEN and will internally be set to
       * dispatch to the correct backend-specific functions when
       * af_set_memory_manager returns.
       */
      /**
       * \returns The id of the active device as indexed by ArrayFire
       *
       * \note ID can be the same for arrays belonging to different backends
       */
      int (*af_memory_manager_get_active_device_id)(af_memory_manager* impl_);

      /**
       * \param[in] id id of the active device as indexed by ArrayFire
       * \returns the amount of available memory
       */
      int (*af_memory_manager_get_max_memory_size)(
        af_memory_manager* impl_,
        int id
      );

      /**
       * Allocates a pointer of a given size via the current
       * backend-specific implemenetation
       *
       * \param[in] bytes the amount of memory to allocate
       * \returns a pointer to allocated memory
       */
      void* (*af_memory_manager_native_alloc)(af_memory_manager* impl_,
                                              size_t size);

      /**
       * Frees a pointer via the current backend-specific implemenetation
       *
       * \param[in] ptr the pointer to free
       */
      void (*af_memory_manager_native_free)(af_memory_manager* impl_,
                                                    void* ptr);

      // \cond
      // An internally-used handle for referencing the C++ wrapper class so as
      // to enable backend-specific allocator functions to be exposed directly
      // to an af_memory_manager.
      af_memory_manager* wrapper_handle;
      // \endcond
    } af_memory_manager;

    /**
     * \brief Sets the pinned memory manager for the current backend.
     *
     * \param[in] manager a pointer to a C-based implementation of
     * af_memory_manager via typesafe C struct inheritance.
     * \param[in] api the api (one of C or C++) used to define the manager
     *
     * \note The existing memory manager for this backend will be destroyed
     * before the new memory manager is set. On ArrayFire shutdown,
     * this memory manager will be freed automatically; its dtor will be
     * called and ArrayFire will free any memory associated with the manager.
     * Prematurely destroying the manager may lead to undefined behavior.
     *
     * \ingroup memory_manager_c_api
     */
    AFAPI af_err af_set_memory_manager(af_memory_manager* manager,
                                       af_memory_manager_api_type api);

    /**
     * \brief Sets the pinned memory manager for the current backend.
     *
     * \param[in] manager a pointer to a C-based implementation of
     * af_memory_manager via typesafe C struct inheritance.
     * \param[in] api the api (one of C or C++) used to define the manager
     *
     * \note The existing memory manager for this backend will be destroyed
     * before the new memory manager is set. On ArrayFire shutdown,
     * this memory manager will be freed automatically; its dtor will be
     * called and ArrayFire will free any memory associated with the manager.
     * Prematurely destroying the manager may lead to undefined behavior.
     * \ingroup memory_manager_c_api
     */
    AFAPI af_err af_set_pinned_memory_manager(af_memory_manager* manager,
                                              af_memory_manager_api_type api);
#endif

#ifdef __cplusplus
}
#endif

// C++ API
#ifdef __cplusplus

namespace spdlog {
class logger;
}

namespace af
{

// FIXME: should this be 37?
#if AF_API_VERSION >= 36
    /// \cond
    /**
     * A generic interface that backend-specific memory managers
     * implement.
     *
     * NB: Overriding or defining methods on a class derived from
     * BackendMemoryManager will have NO EFFECT, since derived
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
         // intentionally protected
        BackendMemoryManager() = default;
        std::shared_ptr<spdlog::logger> logger;
      public:
        virtual ~BackendMemoryManager() = default;
        virtual int getActiveDeviceId() = 0;
        virtual size_t getMaxMemorySize(int id) = 0;
        virtual void* nativeAlloc(const size_t bytes) = 0;
        virtual void nativeFree(void *ptr) = 0;
        virtual spdlog::logger* getLogger() final {
            return this->logger.get();
        }
    };
    /// \endcond

    /**
     * \brief A C++ interface for defining functions for a custom memory
     * manager to be used by ArrayFire.
     *
     * Functions in this interface are called by the rest of the ArrayFire API
     * during operations requiring memory (e.g. allocating Arrays, optimizing
     * allocations for the JIT, etc).
     *
     * All functions marked pure virtual must be defined, although depending
     * on the specific memory manager implementation, some can be noops or
     * throw exceptions (for example, a memory manager that doesn't implement
     * garbage collection might throw when called).
     *
     * ArrayFire defines native, backend and device-specific allocation
     * functions that rely on backend-specific ArrayFire APIs. These
     * can be called from any type deriving from MemoryManagerBase, e.g:
     * \code
       void *alloc(const size_t size, bool user_lock) override {
         // ...
         void* ptr = this->nativeAlloc(...);
         // ...
       }
       \endcode
     *
     * \ingroup memory_manager_cpp_api
     */
    class MemoryManagerBase : public af_memory_manager
    {
     public:
      MemoryManagerBase() = default;
      virtual ~MemoryManagerBase() = default;

      /**
       * \brief Performs any necessary post-constructor initialization for
       * the memory manager.
       *
       * \note Called when \ref setMemoryManager or \ref setPinnedMemoryManager
       * is invoked. Backend-specific methods are callable (e.g. nativeAlloc)
       */
      virtual void initialize() = 0;

      /**
       * \brief Called by the BackendMemoryManager dtor for each device;
       * performs device or backend-specific cleanup during destruction
       *
       * Since the BackendMemoryManager is destroyed before any derived
       * instance of MemoryManagerBase, this is called before
       * ~MemoryManagerBase is invoked.
       */
      virtual void shutdown() = 0;

      /**
       * \brief Adds an external OpenCL device to the memory manager.
       *
       * \param[in] device device id of the new device
       *
       * \note Intended to be used with OpenCL backend, where
       * users are allowed to add external devices(context, device pair)
       * to the list of devices automatically detected by the library
       */
      virtual void addMemoryManagement(int device) = 0;

      /**
       * \brief Removes an external OpenCL device to the memory manager.
       *
       * \param[in] device device id of the new device
       *
       * \note Intended to be used with OpenCL backend, where
       * users are allowed to add external devices(context, device pair)
       * to the list of devices automatically detected by the library
       */
      virtual void removeMemoryManagement(int device) = 0;

      /**
       * \brief Returns a pointer of size at least `size`.
       *
       * \param[in] size the size of the buffer to allocate
       * \param[in] user_lock whether or not the memory was allocated via a
       * call to af::alloc/af_alloc_device or internally by ArrayFire
       */
      virtual void *alloc(const size_t size, bool user_lock) = 0;

      /** \brief Returns the size of the buffer at the pointer allocated by
       * the memory manager.
       *
       * \param[in] ptr pointer to check allocation size
       */
      virtual size_t allocated(void *ptr) = 0;

      /**
       * \brief Frees the pointer (or marks as unused).
       *
       * \note ArrayFire will call this function when memory is no longer needed.
       * Depending on implementation, this can either free memory immediately
       * or mark it for later deletion.
       *
       * \param[in] ptr the pointer to unlock.
       * \param[in] user_unlock whether or not the call to unlock is via the user
       * with af::free/af_free_device or internally by ArrayFire
       */
      virtual void unlock(void *ptr, bool user_unlock) = 0;

      /**
       * \brief Gives information about memory manager state.
       *
       * \note This information is used by the ArrayFire JIT.
       *
       * \param[out] alloc_bytes the number of bytes allocated by the manager
       * \param[out] alloc_buffers the number of buffers created by the manager
       * \param[out] lock_bytes the number of bytes in use
       * \param[out] lock_buffers the number of buffers in use
       */
      virtual void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                              size_t *lock_bytes,  size_t *lock_buffers) = 0;

      /**
       * \brief Locks memory in the manager as user-controlled
       *
       * \note Generally called when ArrayFire returns a pointer to the user,
       * as is the case when acquiring a device pointer
       *
       * \param[in] ptr the pointer to lock
       */
      virtual void userLock(const void *ptr) = 0;

      /**
       * \brief Releases memory back to the manager from the user
       *
       * \param[in] ptr the pointer to release
       */
      virtual void userUnlock(const void *ptr) = 0;

      /**
       * \brief Returns if a piece of memory is locked by the user.
       *
       * \param[in] ptr memory to check
       */
      virtual bool isUserLocked(const void *ptr) = 0;

      /**
       * \brief Returns whether allocated memory is at an
       * implementation-defined limit (generally per-device)
       *
       * \note This is an implementation-specific function used by the JIT
       */
      virtual bool checkMemoryLimit() = 0;

      /**
       * \brief Returns the maximum number of bytes that can be allocated
       * by the manager (generally per-device)
       *
       * \note Used by the JIT
       */
      virtual size_t getMaxBytes() = 0;

      /**
       * \brief Returns the maximum number of buffers that can be allocated
       * by the manager (generally per-device)
       *
       * \note Used by the JIT
       */
      virtual unsigned getMaxBuffers() = 0;

      /**
       * \brief Prints information about current memory manager state
       *
       * \param[in] msg a idenfier string with which the message can
       * be prefixed
       * \param[in] device the device to print status about
       */
      virtual void printInfo(const char *msg, const int device) = 0;

      /**
       * \brief Cleans up unused memory.
       *
       * \note While not all memory manager implementations will support
       * garbage collection, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      virtual void garbageCollect() = 0;

      /**
       * \brief Gets the memory step size of the manager.
       *
       * \note While not all memory manager implementations will support
       * a step size detail, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      virtual size_t getMemStepSize() = 0;

      /**
       * \brief Sets the memory step size of the manager.
       *
       * \param[in] new_step_size the new step size for the memory manager.
       *
       * \note While not all memory manager implementations will support
       * a step size detail, this method must be implemented (the existing
       * ArrayFire API requires it). Throwing an exception/leaving the
       * implementation as a noop is acceptable.
       */
      virtual void setMemStepSize(size_t new_step_size) = 0;

      // \cond
      /// Sets a BackendMemoryManager for this memory manager implementation.
      /// Use internally to support exposing backend-specific functions (below)
      virtual void setBackendManager(
        std::unique_ptr<BackendMemoryManager> manager) final {
          backendMem_ = std::move(manager);
      }
      // \endcond

    protected:
      /// \cond
      /// A backend-specific collection of methods for device-specific
      /// memory allocation facilities including native allocation and frees,
      /// device ID acquisition, and max-memory lookup. Callable by derived
      /// implementations of MemoryManagerBase.
      ///
      /// Created and set on the derived implementation
      /// when `af::setMemoryManager` returns such that calls to these
      /// methods are correctly dispatched to the proper internal ArrayFire
      /// implementation.
      std::unique_ptr<BackendMemoryManager> backendMem_;
      /// \endcond
      
    public:
      /**
       * \returns The id of the active device as indexed by ArrayFire
       *
       * \note ID can be the same for arrays belonging to different backends
       */
      virtual int getActiveDeviceId() final {
          return backendMem_->getActiveDeviceId();
      }

      /**
       * \param[in] id id of the active device as indexed by ArrayFire
       * \returns the amount of available memory
       */
      virtual size_t getMaxMemorySize(int id) final {
          return backendMem_->getMaxMemorySize(id);
      }

      /**
       * Allocates a pointer of a given size via the current
       * backend-specific implemenetation
       *
       * \param[in] bytes the amount of memory to allocate
       * \returns a pointer to allocated memory
       */
      virtual void* nativeAlloc(const size_t bytes) final {
          return backendMem_->nativeAlloc(bytes);
      }

      /**
       * Frees a pointer via the current backend-specific implemenetation
       *
       * \param[in] ptr the pointer to free
       */
      virtual void nativeFree(void *ptr) final {
          backendMem_->nativeFree(ptr);
      }

      /**
       * \returns a spdlog::logger for memory logging
       */
      virtual spdlog::logger* getLogger() final {
          return backendMem_->getLogger();
      }
    };

    /**
     * \brief Sets the memory manager for the current backend.
     *
     * \param[in] manager a pointer to a derived implementation of
     * \ref af::MemoryManagerBase
     *
     * Equivalent to calling  \ref af_set_pinned_memory_manager with
     * \ref AF_CPP_MEMORY_MANAGER_API.
     *
     * \note The existing memory manager for this backend will be destroyed
     * before the new memory manager is set. On ArrayFire shutdown,
     * this memory manager will be freed automatically; its dtor will be
     * called and ArrayFire will free any memory associated with the manager.
     * Prematurely destroying the manager may lead to undefined behavior.
     *
     * \ingroup memory_manager_cpp_api
     */
    AFAPI void setMemoryManager(af_memory_manager* manager);


    /**
     * \brief Sets the pinned device memory manager for the current backend.
     *
     * \param[in] manager a pointer to a derived implementation of
     * \ref af::MemoryManagerBase
     *
     * Equivalent to calling \ref af_set_pinned_memory_manager with
     * \ref AF_CPP_MEMORY_MANAGER_API.
     *
     * \note The existing pinned memory manager for this backend will be
     * destroyed before the new memory manager is set. On ArrayFire shutdown,
     * this memory manager will be freed automatically; its dtor will be
     * called and ArrayFire will free any memory associated with the manager.
     * Prematurely destroying the manager may lead to undefined behavior.
     *
     * \ingroup memory_manager_cpp_api
     */
    AFAPI void setPinnedMemoryManager(af_memory_manager* manager);

#endif

}
#endif
