/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <af/event.h>

#include <stddef.h>

#if AF_API_VERSION >= 37

typedef void* af_buffer_info;

typedef void* af_memory_manager;

#ifdef __cplusplus
namespace af {

/// A simple RAII wrapper for af_buffer_info
class AFAPI buffer_info {
    af_buffer_info p_;

   public:
    buffer_info(af_buffer_info p);
    buffer_info(void* ptr, af_event event);
    ~buffer_info();
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    buffer_info(buffer_info&& other);
    buffer_info& operator=(buffer_info&& other);
#endif
    void* getPtr() const;
    void setPtr(void* ptr);
    af_event getEvent() const;
    void setEvent(af_event event);
    af_buffer_info get() const;
    af_event unlockEvent();
    void* unlockPtr();

   private:
    buffer_info& operator=(const buffer_info& other);
    buffer_info(const buffer_info& other);
};

}  // namespace af
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

AFAPI af_err af_create_buffer_info(af_buffer_info* buf, void* ptr,
                                   af_event event);

/// \brief deletes the \ref af_buffer_info and the resources its tracking
///
///  Deletes the \ref af_buffer_info object and its tracked resources. If buffer
///  still holds
/// the pointer,  that pointer is freed after its associated event has
/// triggered.
///
/// \param[in] buf The af_buffer_info object that will be deleted
/// \returns AF_SUCCESS
AFAPI af_err af_delete_buffer_info(af_buffer_info buf);

AFAPI af_err af_buffer_info_get_ptr(void** ptr, af_buffer_info buf);

AFAPI af_err af_buffer_info_get_event(af_event* event, af_buffer_info buf);

AFAPI af_err af_buffer_info_set_ptr(af_buffer_info buf, void* ptr);

AFAPI af_err af_buffer_info_set_event(af_buffer_info buf, af_event event);

/// \brief Disassociates the \ref af_event from the \ref af_buffer_info object
///
/// Gets the \ref af_event and disassociated it from the af_buffer_info object.
/// Deleting the af_buffer_info object will not affect this event.
///
/// param[out] event The \ref af_event that will be disassociated. If NULL no
/// event is
///                   returned and the event is NOT freed
/// param[in] buf The target \ref af_buffer_info object
/// \returns AF_SUCCESS
AFAPI af_err af_unlock_buffer_info_event(af_event* event, af_buffer_info buf);

/// \brief Disassociates the pointer from the \ref af_buffer_info object
///
/// Gets the pointer and disassociated it from the \ref af_buffer_info object.
/// Deleting the af_buffer_info object will not affect this pointer.
///
/// param[out] event The \ref pointer that will be disassociated. If NULL no
/// pointer is
///                  returned and the data is NOT freed.
/// param[in] buf The target \ref af_buffer_info object
/// \returns AF_SUCCESS
AFAPI af_err af_unlock_buffer_info_ptr(void** ptr, af_buffer_info buf);

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

typedef void (*af_memory_manager_initialize_fn)(af_memory_manager);

typedef void (*af_memory_manager_shutdown_fn)(af_memory_manager);

typedef af_buffer_info (*af_memory_manager_alloc_fn)(af_memory_manager, size_t,
                                                     /* bool */ int);

typedef size_t (*af_memory_manager_allocated_fn)(af_memory_manager, void*);

typedef void (*af_memory_manager_unlock_fn)(af_memory_manager, void*, af_event,
                                            /* bool */ int);

typedef void (*af_memory_manager_garbage_collect_fn)(af_memory_manager);

typedef void (*af_memory_manager_print_info_fn)(af_memory_manager, char*, int);

typedef void (*af_memory_manager_usage_info_fn)(af_memory_manager, size_t*,
                                                size_t*, size_t*, size_t*);

typedef void (*af_memory_manager_user_lock_fn)(af_memory_manager, void*);

typedef void (*af_memory_manager_user_unlock_fn)(af_memory_manager, void*);

typedef int (*af_memory_manager_is_user_locked_fn)(af_memory_manager, void*);

typedef size_t (*af_memory_manager_get_mem_step_size_fn)(af_memory_manager);

typedef size_t (*af_memory_manager_get_max_bytes_fn)(af_memory_manager);

typedef unsigned (*af_memory_manager_get_max_buffers_fn)(af_memory_manager);

typedef void (*af_memory_manager_set_mem_step_size_fn)(af_memory_manager,
                                                       size_t);

typedef int (*af_memory_manager_check_memory_limit)(af_memory_manager);

typedef void (*af_memory_manager_add_memory_management)(af_memory_manager, int);

typedef void (*af_memory_manager_remove_memory_management)(af_memory_manager,
                                                           int);

/// \brief Creates a handle to an af_memory_manager
///
/// Creates a blank af_memory_manager with no attached function pointers.
///
/// param[out] out \ref af_memory_manager
/// \returns AF_SUCCESS
AFAPI af_err af_create_memory_manager(af_memory_manager* out);

/// \brief Sets the internal AF memory manager to use the given \ref
/// af_memory_manager
///
/// Creates a blank af_memory_manager with no attached function pointers.
///
/// param[in] handle the \ref af_memory_manager handle to be destroyed
/// \returns AF_SUCCESS
AFAPI af_err af_release_memory_manager(af_memory_manager handle);

AFAPI af_err af_release_memory_manager_pinned(af_memory_manager handle);

/// \brief Creates a handle to an af_memory_manager
///
/// Registers the given memory manager as the AF memory manager - if the default
/// memory manager is current, destroys it and frees resources; if another
/// memory manager is set, does NOT destroy its handle or free associated memory
/// - this must be done manually.
///
/// param[out] out \ref af_memory_manager
/// \returns AF_SUCCESS
AFAPI af_err af_set_memory_manager(af_memory_manager handle);

AFAPI af_err af_set_memory_manager_pinned(af_memory_manager handle);

AFAPI af_err af_memory_manager_get_payload(af_memory_manager handle,
                                           void** payload);

AFAPI af_err af_memory_manager_set_payload(af_memory_manager handle,
                                           void* payload);

AFAPI af_err af_memory_manager_set_initialize_fn(
    af_memory_manager handle, af_memory_manager_initialize_fn fn);
AFAPI af_err af_memory_manager_set_shutdown_fn(
    af_memory_manager handle, af_memory_manager_shutdown_fn fn);
AFAPI af_err af_memory_manager_set_alloc_fn(af_memory_manager handle,
                                            af_memory_manager_alloc_fn fn);

AFAPI af_err af_memory_manager_set_allocated_fn(
    af_memory_manager handle, af_memory_manager_allocated_fn fn);
AFAPI af_err af_memory_manager_set_unlock_fn(af_memory_manager handle,
                                             af_memory_manager_unlock_fn fn);

AFAPI af_err af_memory_manager_set_garbage_collect_fn(
    af_memory_manager handle, af_memory_manager_garbage_collect_fn fn);
AFAPI af_err af_memory_manager_set_print_info_fn(
    af_memory_manager handle, af_memory_manager_print_info_fn fn);
AFAPI af_err af_memory_manager_set_usage_info_fn(
    af_memory_manager handle, af_memory_manager_usage_info_fn fn);

AFAPI af_err af_memory_manager_set_user_lock_fn(
    af_memory_manager handle, af_memory_manager_user_lock_fn fn);
AFAPI af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn);
AFAPI af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn);
AFAPI af_err af_memory_manager_set_get_mem_step_size_fn(
    af_memory_manager handle, af_memory_manager_get_mem_step_size_fn fn);
AFAPI af_err af_memory_manager_set_get_max_bytes_fn(
    af_memory_manager handle, af_memory_manager_get_max_bytes_fn fn);
AFAPI af_err af_memory_manager_set_get_max_buffers_fn(
    af_memory_manager handle, af_memory_manager_get_max_buffers_fn fn);
AFAPI af_err af_memory_manager_set_set_mem_step_size_fn(
    af_memory_manager handle, af_memory_manager_set_mem_step_size_fn fn);

AFAPI af_err af_memory_manager_set_check_memory_limit_fn(
    af_memory_manager handle, af_memory_manager_check_memory_limit fn);
AFAPI af_err af_memory_manager_set_add_memory_management_fn(
    af_memory_manager handle, af_memory_manager_add_memory_management fn);
AFAPI af_err af_memory_manager_set_remove_memory_management_fn(
    af_memory_manager handle, af_memory_manager_remove_memory_management fn);

////////////////////////////////////////////////////////////////////////////////
// Native memory interface functions
AFAPI af_err af_memory_manager_get_active_device_id(af_memory_manager handle,
                                                    int* id);

AFAPI af_err af_memory_manager_native_alloc(af_memory_manager handle,
                                            void** ptr, size_t size);

AFAPI af_err af_memory_manager_native_free(af_memory_manager handle, void* ptr);

AFAPI af_err af_memory_manager_get_max_memory_size(af_memory_manager handle,
                                                   size_t* size, int id);

#ifdef __cplusplus
}
#endif  // __cplusplus

#ifdef __cplusplus
namespace af {

class AFAPI memory_manager {
    af_memory_manager m_;

    typedef af_memory_manager_initialize_fn InitializeFn;
    typedef af_memory_manager_shutdown_fn ShutdownFn;
    typedef af_memory_manager_alloc_fn AllocFn;
    typedef af_memory_manager_allocated_fn AllocatedFn;
    typedef af_memory_manager_unlock_fn UnlockFn;
    typedef af_memory_manager_garbage_collect_fn GarbageCollectFn;
    typedef af_memory_manager_print_info_fn PrintInfoFn;
    typedef af_memory_manager_usage_info_fn UsageInfoFn;
    typedef af_memory_manager_user_lock_fn UserLockFn;
    typedef af_memory_manager_user_unlock_fn UserUnlockFn;
    typedef af_memory_manager_is_user_locked_fn IsUserLockedFn;
    typedef af_memory_manager_get_mem_step_size_fn GetMemStepSizeFn;
    typedef af_memory_manager_get_max_bytes_fn GetMaxBytesFn;
    typedef af_memory_manager_get_max_buffers_fn GetMaxBuffersFn;
    typedef af_memory_manager_set_mem_step_size_fn SetMemStepSizeFn;
    typedef af_memory_manager_check_memory_limit CheckMemoryLimitFn;
    typedef af_memory_manager_add_memory_management AddMemoryManagementFn;
    typedef af_memory_manager_remove_memory_management RemoveMemoryManagementFn;

   public:
    memory_manager();
    memory_manager(af_memory_manager p);
    ~memory_manager();
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    memory_manager(memory_manager&& other);
    memory_manager& operator=(memory_manager&& other);
#endif
    af_memory_manager get() const;

    void registerInitialize(InitializeFn fn);
    void registerShutdown(ShutdownFn fn);
    void registerAlloc(AllocFn fn);
    void registerAllocated(AllocatedFn fn);
    void registerUnlock(UnlockFn fn);
    void registerGarbageCollect(GarbageCollectFn fn);
    void registerPrintInfo(PrintInfoFn fn);
    void registerUsageInfo(UsageInfoFn fn);
    void registerUserLock(UserLockFn fn);
    void registerUserUnlock(UserUnlockFn fn);
    void registerIsUserLocked(IsUserLockedFn fn);
    void registerGetMemStepSize(GetMemStepSizeFn fn);
    void registerGetMaxBytes(GetMaxBytesFn fn);
    void registerGetMaxBuffers(GetMaxBuffersFn fn);
    void registerSetMemStepSize(SetMemStepSizeFn fn);
    void registerCheckMemoryLimit(CheckMemoryLimitFn fn);
    void registerAddMemoryManagement(AddMemoryManagementFn fn);
    void registerRemoveMemoryManagement(RemoveMemoryManagementFn fn);

    void setPayload(void* payload);
    void* getPayload() const;

};  // namespace af

}  // namespace af
#endif  // __cplusplus

#endif  // AF_API_VERSION >= 37
