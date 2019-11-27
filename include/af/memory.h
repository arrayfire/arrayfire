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

/**
   \brief Creates an \ref af_buffer_info handle from an \ref af_event event and
   ptr

   \param[in] buf The \ref af_buffer_info object to be created
   \param[in] ptr A pointer
   \param[in] event An \ref af_event
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_create_buffer_info(af_buffer_info* buf, void* ptr,
                                   af_event event);

/**
   \brief deletes the \ref af_buffer_info and its resources

   Deletes the \ref af_buffer_info object and its tracked resources. If buffer
   still holds a pointer, that pointer is freed. Does NOT enqueue a wait on the
   associated event

   \param[in] buf The \ref af_buffer_info object that will be deleted
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_delete_buffer_info(af_buffer_info buf);

/**
   \brief Retrieves a pointer from an \ref af_buffer_info

   \param[out] ptr The associated pointer
   \param[in] buf The \ref af_buffer_info object
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_buffer_info_get_ptr(void** ptr, af_buffer_info buf);

/**
   \brief Retrieves an \ref af_event from an \ref af_buffer_info

   \param[out] event The associated event
   \param[in] buf The \ref af_buffer_info object
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_buffer_info_get_event(af_event* event, af_buffer_info buf);

/**
   \brief Sets a pointer on an \ref af_buffer_info

   \param[in] buf The \ref af_buffer_info object
   \param[in] ptr The pointer to set
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_buffer_info_set_ptr(af_buffer_info buf, void* ptr);

/**
   \brief Sets an \ref af_event on an \ref af_buffer_info

   \param[in] buf The \ref af_buffer_info object
   \param[in] event The \ref af_event to set
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_buffer_info_set_event(af_buffer_info buf, af_event event);

/**
   \brief Disassociates the \ref af_event from the \ref af_buffer_info object

   Gets the \ref af_event and disassociated it from the af_buffer_info object.
   Deleting the af_buffer_info object will not affect this event.

   \param[out] event The \ref af_event that will be disassociated. If NULL no
   event is returned and the event is NOT freed
   \param[in] buf The target \ref af_buffer_info object
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_unlock_buffer_info_event(af_event* event, af_buffer_info buf);

/**
   \brief Disassociates the pointer from the \ref af_buffer_info object

   Gets the pointer and disassociated it from the \ref af_buffer_info object.
   Deleting the \ref af_buffer_info object will not affect this pointer.

   \param[out] ptr The pointer that will be disassociated. If NULL no
   pointer is returned and the data is NOT freed.
   \param[in] buf The target \ref af_buffer_info object
   \returns AF_SUCCESS

   \ingroup buffer_info
*/
AFAPI af_err af_unlock_buffer_info_ptr(void** ptr, af_buffer_info buf);

/**
   \brief Called after a memory manager is set and becomes active.

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_initialize_fn)(af_memory_manager handle);

/**
   \brief Called after a memory manager is unset and becomes unused

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_shutdown_fn)(af_memory_manager handle);

/**
   \brief Called when memory is to be allocated by ArrayFire

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] buffer_info a pointer to a \ref af_buffer_info containing the
   pointer to the allocated buffer and an associated \ref af_event
   \param[in] bytes number of bytes to allocate
   \param[in] user_lock a truthy value corresponding to whether or not the
   memory should have a user lock associated with it
   \param[in] ndims the number of dimensions associated with the allocated
   memory. This value is currently always 1
   \param[in] dims a \ref dim_t containing the dimensions of the allocation by
   number of elements
   \param[in] element_size the number of bytes per element of allocated memory

   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_alloc_fn)(
    af_memory_manager handle, af_buffer_info* buffer_info, size_t bytes,
    /* bool */ int user_lock, const unsigned ndims, const dim_t* const dims,
    const unsigned element_size);

/**
   \brief Checks the amount of allocated memory for a pointer

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] size the size of the allocated memory for the pointer
   \param[in] ptr the pointer to query
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_allocated_fn)(af_memory_manager handle,
                                                 size_t* size, void* ptr);

/**
   \brief Unlocks memory from use

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] ptr the pointer to query
   \param[in] event a new \ref af_event which will be marked before the free is
   executed such that enqueing a wait on this event
   \param[in] user_unlock frees the memory from user lock
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_unlock_fn)(af_memory_manager handle,
                                              void* ptr, af_event event,
                                              /* bool */ int user_unlock);

/**
   \brief Called to signal the memory manager should free memory if possible

   Called by some external functions that allocate their own memory if they
   receive an out of memory in order to free up other memory on a device

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_signal_memory_cleanup_fn)(
    af_memory_manager handle);

/**
   \brief Populates memory information into a passed buffer

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] a buffer to which a message will be populated
   \param[in] the device id for which to print memory
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_print_info_fn)(af_memory_manager handle,
                                                  char*, int);

/**
   \brief Called to lock a buffer as user-owned memory

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[in] ptr pointer to the buffer to lock
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_user_lock_fn)(af_memory_manager handle,
                                                 void* ptr);

/**
   \brief Called to unlock a buffer from user-owned memory

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[in] ptr pointer to the buffer to unlock
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_user_unlock_fn)(af_memory_manager handle,
                                                   void* ptr);

/**
   \brief Queries if a buffer is user locked

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] out a truthy value corresponding to if the buffer is user locked
   \param[in] ptr pointer to the buffer to query
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_is_user_locked_fn)(af_memory_manager handle,
                                                      int* out, void* ptr);

/**
   \brief Gets memory pressure for a memory manager

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] pressure the memory pressure value
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_get_memory_pressure_fn)(af_memory_manager,
                                                           float* pressure);

/**
   \brief Called to query if additions to the JIT tree would exert too much
   memory pressure

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[out] out a truthy value if too much memory pressure is exerted
   \param[in] size the size of the allocation required for the new JIT nodes
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef af_err (*af_memory_manager_jit_tree_exceeds_memory_pressure_fn)(
    af_memory_manager handle, int* out, size_t size);

/**
   \brief Adds a new device to the memory manager (OpenCL only)

   \param[in] handle a pointer to the active \ref af_memory_manager handle
   \param[in] id the id of the device to add
   \returns AF_SUCCESS

   \ingroup memory_manager_api
*/
typedef void (*af_memory_manager_add_memory_management)(
    af_memory_manager handle, int id);

/**
    \brief Removes a device from the memory manager (OpenCL only)

    \param[in] handle a pointer to the active \ref af_memory_manager handle
    \param[in] id the id of the device to remove
    \returns AF_SUCCESS

    \ingroup memory_manager_api
*/
typedef void (*af_memory_manager_remove_memory_management)(af_memory_manager,
                                                           int id);

/**
   \brief Creates an \ref af_memory_manager handle

   Creates a blank af_memory_manager with no attached function pointers.

   \param[out] out \ref af_memory_manager
   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_create_memory_manager(af_memory_manager* out);

/**
   \brief Destroys an \ref af_memory_manager handle.

   Destroys a memory manager handle, does NOT call the
   \ref af_memory_manager_shutdown_fn associated with the af_memory_manager.

   \param[in] handle the \ref af_memory_manager handle to be destroyed
   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_release_memory_manager(af_memory_manager handle);

/**
   \brief Sets an af_memory_manager to be the default memory manager for
   non-pinned memory allocations in ArrayFire.

   Registers the given memory manager as the AF memory manager non-pinned
   memory allocations - does NOT shut down or release the existing memory
   manager or free any associated memory.

   \param[in] handle the \ref af_memory_manager handle to use
   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_set_memory_manager(af_memory_manager handle);

/**
   \brief Sets an af_memory_manager to be the default memory manager for
   pinned memory allocations in ArrayFire.

   Registers the given memory manager as the AF memory manager for pinned
   memory allocations - does NOT shut down or release the existing memory
   manager or free any associated memory.

   \param[in] handle the \ref af_memory_manager handle to use
   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_set_memory_manager_pinned(af_memory_manager handle);

/**
   \brief Reset the memory manager being used in ArrayFire to the default
   memory manager, shutting down the existing memory manager.

   Calls the associated af_memory_manager_shutdown_fn on
   the existing memory manager. If the default memory manager is set,
   ALL associated memory will be freed on shutdown. Custom behavior that
   does not free all memory can be defined for a custom memory manager
   as per the specific implementation of its associated
   af_memory_manager_shutdown_fn.

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_unset_memory_manager();

/**
   \brief Reset the pinned memory manager being used in ArrayFire to the
   default memory manager, shutting down the existing pinned memory manager.

   Calls the associated af_memory_manager_shutdown_fn on
   the existing pinned memory manager. If the default memory manager is set,
   ALL associated pinned memory will be freed on shutdown. Custom behavior that
   does not free all pinned memory can be defined for a custom memory manager
   as per the specific implementation of its associated
   af_memory_manager_shutdown_fn.

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_unset_memory_manager_pinned();

/**
   \brief Gets the payload ptr from an \ref af_memory_manager

   \param[in] handle the \ref af_memory_manager handle
   \param[out] payload pointer to the payload pointer

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_get_payload(af_memory_manager handle,
                                           void** payload);

/**
   \brief Gets the payload ptr from an \ref af_memory_manager

   \param[in] handle the \ref af_memory_manager handle
   \param[out] payload pointer to the payload pointer

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_payload(af_memory_manager handle,
                                           void* payload);

/**
   \brief Sets an \ref af_memory_manager_initialize_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_initialize_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_initialize_fn(
    af_memory_manager handle, af_memory_manager_initialize_fn fn);

/**
   \brief Sets an \ref af_memory_manager_shutdown_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_shutdown_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_shutdown_fn(
    af_memory_manager handle, af_memory_manager_shutdown_fn fn);

/**
   \brief Sets an \ref af_memory_manager_alloc_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_alloc_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_alloc_fn(af_memory_manager handle,
                                            af_memory_manager_alloc_fn fn);

/**
   \brief Sets an \ref af_memory_manager_allocated_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_allocated_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_allocated_fn(
    af_memory_manager handle, af_memory_manager_allocated_fn fn);

/**
   \brief Sets an \ref af_memory_manager_unlock_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_unlock_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_unlock_fn(af_memory_manager handle,
                                             af_memory_manager_unlock_fn fn);

/**
   \brief Sets an \ref af_memory_manager_signal_memory_cleanup_fn for a memory
   manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_signal_memory_cleanup_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_signal_memory_cleanup_fn(
    af_memory_manager handle, af_memory_manager_signal_memory_cleanup_fn fn);

/**
   \brief Sets an \ref af_memory_manager_print_info_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_print_info_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_print_info_fn(
    af_memory_manager handle, af_memory_manager_print_info_fn fn);

/**
   \brief Sets an \ref af_memory_manager_user_lock_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_user_lock_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_user_lock_fn(
    af_memory_manager handle, af_memory_manager_user_lock_fn fn);

/**
   \brief Sets an \ref af_memory_manager_user_unlock_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_user_unlock_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn);

/**
   \brief Sets an \ref af_memory_manager_is_user_locked_fn for a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_is_user_locked_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn);

/**
   \brief Sets an \ref af_memory_manager_get_memory_pressure_fn for a memory
   manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_get_memory_pressure_fn to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_get_memory_pressure_fn(
    af_memory_manager handle, af_memory_manager_get_memory_pressure_fn fn);

/**
   \brief Sets an \ref af_memory_manager_jit_tree_exceeds_memory_pressure_fn for
   a memory manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_jit_tree_exceeds_memory_pressure_fn
   to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    af_memory_manager handle,
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn fn);

/**
   \brief Sets an \ref af_memory_manager_add_memory_management for a memory
   manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_add_memory_management to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_add_memory_management_fn(
    af_memory_manager handle, af_memory_manager_add_memory_management fn);

/**
   \brief Sets an \ref af_memory_manager_remove_memory_management for a memory
   manager

   \param[in] handle the \ref af_memory_manager handle
   \param[in] fn the \ref af_memory_manager_remove_memory_management to set

   \returns AF_SUCCESS
   \ingroup memory_manager_utils
*/
AFAPI af_err af_memory_manager_set_remove_memory_management_fn(
    af_memory_manager handle, af_memory_manager_remove_memory_management fn);

////////////////// Native memory interface functions

/**
   \brief Gets the id of the currently-active device

   \param[in] handle the \ref af_memory_manager handle
   \param[out] id the id of the active device

   \returns AF_SUCCESS
   \ingroup native_memory_interface
*/
AFAPI af_err af_memory_manager_get_active_device_id(af_memory_manager handle,
                                                    int* id);

/**
   \brief Allocates memory with a native memory function for the active backend

   \param[in] handle the \ref af_memory_manager handle
   \param[out] ptr the pointer to free
   \param[in] size the size of the pointer allocation

   \returns AF_SUCCESS
   \ingroup native_memory_interface
*/
AFAPI af_err af_memory_manager_native_alloc(af_memory_manager handle,
                                            void** ptr, size_t size);

/**
    \brief Frees a pointer with a native memory function for the active backend

    \param[in] handle the \ref af_memory_manager handle
    \param[in] ptr the pointer to free

    \returns AF_SUCCESS
    \ingroup native_memory_interface
*/
AFAPI af_err af_memory_manager_native_free(af_memory_manager handle, void* ptr);

/** \brief Gets the maximum memory size for a managed device.

  \param[in] handle the \ref af_memory_manager handle
  \param[out] size the max memory size for the device
  \param[in] id the device id

  \returns AF_SUCCESS
  \ingroup native_memory_interface */
AFAPI af_err af_memory_manager_get_max_memory_size(af_memory_manager handle,
                                                   size_t* size, int id);

/**
\brief Gets the memory pressure threshold for a memory manager.

  \param[in] handle the \ref af_memory_manager handle
  \param[out] value the memory pressure threshold

  \returns AF_SUCCESS
  \ingroup native_memory_interface
*/
AFAPI af_err af_memory_manager_get_memory_pressure_threshold(
    af_memory_manager handle, float* value);

/**
    \brief Sets the memory pressure threshold for a memory manager.

    The memory pressure threshold determines when the JIT tree evaluates based
    on how much memory usage there is. If the value returned by \ref
    af_memory_manager_get_memory_pressure_fn exceeds the memory pressure
    threshold, the JIT will evaluate a subtree if generated kernels are valid.

    \param[in] handle the \ref af_memory_manager handle
    \param[in] value the new threshold value

    \returns AF_SUCCESS
    \ingroup native_memory_interface
*/
AFAPI af_err af_memory_manager_set_memory_pressure_threshold(
    af_memory_manager handle, float value);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // AF_API_VERSION >= 37
