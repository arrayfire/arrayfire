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

typedef af_err (*af_memory_manager_initialize_fn)(af_memory_manager);

typedef af_err (*af_memory_manager_shutdown_fn)(af_memory_manager);

typedef af_err (*af_memory_manager_alloc_fn)(af_memory_manager, af_buffer_info*,
                                             size_t,
                                             /* bool */ int);

typedef af_err (*af_memory_manager_allocated_fn)(af_memory_manager, size_t*,
                                                 void*);

typedef af_err (*af_memory_manager_unlock_fn)(af_memory_manager, void*,
                                              af_event,
                                              /* bool */ int);

typedef af_err (*af_memory_manager_signal_memory_cleanup_fn)(af_memory_manager);

typedef af_err (*af_memory_manager_print_info_fn)(af_memory_manager, char*,
                                                  int);

typedef af_err (*af_memory_manager_user_lock_fn)(af_memory_manager, void*);

typedef af_err (*af_memory_manager_user_unlock_fn)(af_memory_manager, void*);

typedef af_err (*af_memory_manager_is_user_locked_fn)(af_memory_manager, int*,
                                                      void*);

typedef af_err (*af_memory_manager_get_memory_pressure_fn)(af_memory_manager,
                                                           float*);

typedef af_err (*af_memory_manager_jit_tree_exceeds_memory_pressure_fn)(
    af_memory_manager, int*, size_t);

typedef void (*af_memory_manager_add_memory_management)(af_memory_manager, int);

typedef void (*af_memory_manager_remove_memory_management)(af_memory_manager,
                                                           int);

/// \brief Creates an \ref af_memory_manager handle
///
/// Creates a blank af_memory_manager with no attached function pointers.
///
/// param[in] out \ref af_memory_manager
/// \returns AF_SUCCESS
AFAPI af_err af_create_memory_manager(af_memory_manager* out);

/// \brief Destroys an \ref af_memory_manager handle.
///
/// Destroys a memory manager handle, does NOT call the
/// af_memory_manager_shutdown_fn associated with the af_memory_manager.
///
/// param[in] handle the \ref af_memory_manager handle to be destroyed
/// \returns AF_SUCCESS
AFAPI af_err af_release_memory_manager(af_memory_manager handle);

/// \brief Sets an af_memory_manager to be the default memory manager for
/// non-pinned memory allocations in ArrayFire.
///
/// Registers the given memory manager as the AF memory manager non-pinned
/// memory allocations - does NOT shut down or release the existing memory
/// manager or free any associated memory.
///
/// param[in] in \ref af_memory_manager
/// \returns AF_SUCCESS
AFAPI af_err af_set_memory_manager(af_memory_manager handle);

/// \brief Sets an af_memory_manager to be the default memory manager for
/// pinned memory allocations in ArrayFire.
///
/// Registers the given memory manager as the AF memory manager for pinned
/// memory allocations - does NOT shut down or release the existing memory
/// manager or free any associated memory.
///
/// param[in] in \ref af_memory_manager
/// \returns AF_SUCCESS
AFAPI af_err af_set_memory_manager_pinned(af_memory_manager handle);

/// \brief Reset the memory manager being used in ArrayFire to the default
/// memory manager, shutting down the existing memory manager.
///
/// Calls the associated af_memory_manager_shutdown_fn on
/// the existing memory manager. If the default memory manager is set,
/// ALL associated memory will be freed on shutdown. Custom behavior that
/// does not free all memory can be defined for a custom memory manager
/// as per the specific implementation of its associated
/// af_memory_manager_shutdown_fn.
///
/// \returns AF_SUCCESS
AFAPI af_err af_unset_memory_manager();

/// \brief Reset the pinned memory manager being used in ArrayFire to the
/// default memory manager, shutting down the existing pinned memory manager.
///
/// Calls the associated af_memory_manager_shutdown_fn on
/// the existing pinned memory manager. If the default memory manager is set,
/// ALL associated pinned memory will be freed on shutdown. Custom behavior that
/// does not free all pinned memory can be defined for a custom memory manager
/// as per the specific implementation of its associated
/// af_memory_manager_shutdown_fn.
///
/// \returns AF_SUCCESS
AFAPI af_err af_unset_memory_manager_pinned();

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

AFAPI af_err af_memory_manager_set_signal_memory_cleanup_fn(
    af_memory_manager handle, af_memory_manager_signal_memory_cleanup_fn fn);
AFAPI af_err af_memory_manager_set_print_info_fn(
    af_memory_manager handle, af_memory_manager_print_info_fn fn);

AFAPI af_err af_memory_manager_set_user_lock_fn(
    af_memory_manager handle, af_memory_manager_user_lock_fn fn);
AFAPI af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn);
AFAPI af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn);

AFAPI af_err af_memory_manager_set_get_memory_pressure_fn(
    af_memory_manager handle, af_memory_manager_get_memory_pressure_fn fn);
AFAPI af_err af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    af_memory_manager handle,
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn fn);

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

AFAPI af_err af_memory_manager_get_memory_pressure_threshold(
    af_memory_manager handle, float* value);

AFAPI af_err af_memory_manager_set_memory_pressure_threshold(
    af_memory_manager handle, float value);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // AF_API_VERSION >= 37
