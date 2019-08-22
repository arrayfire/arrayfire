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

#ifdef __cplusplus
namespace af
{
    /**
       \defgroup device_func_info info

       Display ArrayFire and device info

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void info();
    /**
       @}
    */

    /**
       \defgroup device_func_info_string infoString

       Get af::info() as a string

       @{

       \brief Returns the output of af::info() as a string

       \param[in] verbose flag to return verbose info

       \returns string containing output of af::info()

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI const char* infoString(const bool verbose = false);
    /**
       @}
    */

    /**
        \copydoc device_func_prop

        \ingroup device_func_prop
    */
    AFAPI void deviceInfo(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    /// \brief Gets the number of devices
    ///
    /// \copydoc device_func_count
    /// \returns the number of devices on the system
    /// \ingroup device_func_count
    AFAPI int getDeviceCount();

    /// \brief Gets the current device ID
    ///
    /// \copydoc device_func_get
    /// \returns the device ID of the current device
    /// \ingroup device_func_get
    AFAPI int getDevice();

    /// \brief Queries the current device for double precision floating point
    ///        support
    ///
    /// \param[in] device the ID of the device to query
    ///
    /// \returns true if the \p device supports double precision operations.
    ///          false otherwise
    /// \ingroup device_func_dbl
    AFAPI bool isDoubleAvailable(const int device);

    /// \brief Queries the current device for half precision floating point
    ///        support
    ///
    /// \param[in] device the ID of the device to query
    ///
    /// \returns true if the \p device supports half precision operations.
    ///          false otherwise
    /// \ingroup device_func_half
    AFAPI bool isHalfAvailable(const int device);

    /// \brief Sets the current device
    ///
    /// \param[in] device The ID of the target device
    /// \ingroup device_func_set
    AFAPI void setDevice(const int device);

    /// \brief Blocks until the \p device is finished processing
    ///
    /// \param[in] device is the target device
    /// \ingroup device_func_sync
    AFAPI void sync(const int device = -1);

    /// \ingroup device_func_alloc
    /// @{
    /// \brief Allocates memory using ArrayFire's memory manager
    ///
    /// \copydoc device_func_alloc
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \note The device memory returned by this function is only freed if af::free() is called explicitly

    AFAPI void *alloc(const size_t elements, const dtype type);

    /// \brief Allocates memory using ArrayFire's memory manager
    //
    /// \copydoc device_func_alloc
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \note the size of the memory allocated is the number of \p elements *
    ///         sizeof(type)
    ///
    /// \note The device memory returned by this function is only freed if af::free() is called explicitly
    template<typename T>
    T* alloc(const size_t elements);
    /// @}

    /// \ingroup device_func_free
    ///
    /// \copydoc device_func_free
    /// \param[in] ptr the memory to free
    ///
    /// This function will free a device pointer even if it has been previously locked.
    AFAPI void free(const void *ptr);

    /// \ingroup device_func_pinned
    /// @{
    ///
    /// \copydoc device_func_pinned
    ///
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns the pointer to the memory
    AFAPI void *pinned(const size_t elements, const dtype type);

    /// \copydoc device_func_pinned
    ///
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    template<typename T>
    T* pinned(const size_t elements);
    /// @}

    /// \ingroup device_func_free_pinned
    ///
    /// \copydoc device_func_free_pinned
    /// \param[in] ptr the memory to free
    AFAPI void freePinned(const void *ptr);

#if AF_API_VERSION >= 33
    /// \brief Allocate memory on host
    ///
    /// \copydoc device_func_alloc_host
    ///
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \ingroup device_func_alloc_host
    AFAPI void *allocHost(const size_t elements, const dtype type);
#endif

#if AF_API_VERSION >= 33
    /// \brief Allocate memory on host
    ///
    /// \copydoc device_func_alloc_host
    ///
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \note the size of the memory allocated is the number of \p elements *
    ///         sizeof(type)
    ///
    /// \ingroup device_func_alloc_host
    template<typename T>
    AFAPI T* allocHost(const size_t elements);
#endif

#if AF_API_VERSION >= 33
    /// \brief Free memory allocated internally by ArrayFire
    //
    /// \copydoc device_func_free_host
    ///
    /// \param[in] ptr the memory to free
    ///
    /// \ingroup device_func_free_host
    AFAPI void freeHost(const void *ptr);
#endif

    /// \ingroup device_func_mem
    /// @{
    /// \brief Gets information about the memory manager
    ///
    /// \param[out] alloc_bytes the number of bytes allocated by the memory
    //                          manager
    /// \param[out] alloc_buffers   the number of buffers created by the memory
    //                              manager
    /// \param[out] lock_bytes The number of bytes in use
    /// \param[out] lock_buffers The number of buffers in use
    ///
    /// \note This function performs a synchronization operation
    AFAPI void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                             size_t *lock_bytes, size_t *lock_buffers);

#if AF_API_VERSION >= 33
    ///
    /// Prints buffer details from the ArrayFire Device Manager
    //
    /// \param [in] msg A message to print before the table
    /// \param [in] device_id print the memory info of the specified device.
    ///  -1 signifies active device.
    //
    /// \ingroup device_func_mem
    ///
    /// \note This function performs a synchronization operation
    AFAPI void printMemInfo(const char *msg = NULL, const int device_id = -1);
#endif

    /// \brief Call the garbage collection function in the memory manager
    ///
    /// \ingroup device_func_mem
    AFAPI void deviceGC();
    /// @}

    /// \brief Set the resolution of memory chunks
    ///
    /// \ingroup device_func_mem
    AFAPI void setMemStepSize(const size_t size);

    /// \brief Get the resolution of memory chunks
    ///
    /// \ingroup device_func_mem
    AFAPI size_t getMemStepSize();
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       \ingroup device_func_info
    */
    AFAPI af_err af_info();

    /**
       \ingroup device_func_info
    */
    AFAPI af_err af_init();

    /**
       \brief Gets the output of af_info() as a string

       \param[out] str contains the string
       \param[in] verbose flag to return verbose info

       \ingroup device_func_info_string
    */
    AFAPI af_err af_info_string(char** str, const bool verbose);

    /**
        \copydoc device_func_prop

        \ingroup device_func_prop
    */
    AFAPI af_err af_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    /**
       \ingroup device_func_count
    */
    AFAPI af_err af_get_device_count(int *num_of_devices);

    /**
       \ingroup device_func_dbl
    */
    AFAPI af_err af_get_dbl_support(bool* available, const int device);

    /**
       \ingroup device_func_half
    */
    AFAPI af_err af_get_half_support(bool *available, const int device);

    /**
       \ingroup device_func_set
    */
    AFAPI af_err af_set_device(const int device);

    /**
       \ingroup device_func_set
    */
    AFAPI af_err af_get_device(int *device);

    /**
       \ingroup device_func_sync
    */
    AFAPI af_err af_sync(const int device);

    /**
       \ingroup device_func_alloc

       This device memory returned by this function can only be freed using
       af_free_device
    */
    AFAPI af_err af_alloc_device(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_free

       This function will free a device pointer even if it has been previously
       locked.
    */
    AFAPI af_err af_free_device(void *ptr);

    /**
       \ingroup device_func_pinned
    */
    AFAPI af_err af_alloc_pinned(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_free_pinned
    */
    AFAPI af_err af_free_pinned(void *ptr);

#if AF_API_VERSION >= 33
    /**
       \ingroup device_func_alloc_host
    */
    AFAPI af_err af_alloc_host(void **ptr, const dim_t bytes);
#endif

#if AF_API_VERSION >= 33
    /**
       \ingroup device_func_free_host
    */
    AFAPI af_err af_free_host(void *ptr);
#endif

    /**
       Create array from device memory
       \ingroup construct_mat
    */
    AFAPI af_err af_device_array(af_array *arr, void *data, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       Get memory information from the memory manager
       \ingroup device_func_mem
    */
    AFAPI af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                                    size_t *lock_bytes, size_t *lock_buffers);

#if AF_API_VERSION >= 33
    /**
       Prints buffer details from the ArrayFire Device Manager.

       The result is a table with several columns:

        * POINTER:   The hex address of the array's device or pinned-memory
                     pointer (depending on the backend)
        * SIZE:      Human-readable size of the array
        * AF LOCK:   Indicates whether ArrayFire is using this chunk of memory.
                     If not, the chunk is ready for reuse.
        * USER LOCK: If set, ArrayFire is prevented from freeing this memory.
                     The chunk is not ready for re-use even if all ArrayFire's
                     references to it go out of scope.

       \param [in] msg A message to print before the table
       \param [in] device_id print the memory info of the specified device.
       -1 signifies active device.
      
       \returns AF_SUCCESS if successful
      
       \ingroup device_func_mem
    */
    AFAPI af_err af_print_mem_info(const char *msg, const int device_id);
#endif

    /**
       Call the garbage collection routine
       \ingroup device_func_mem
    */
    AFAPI af_err af_device_gc();

    /**
       Set the minimum memory chunk size
       \ingroup device_func_mem
    */
    AFAPI af_err af_set_mem_step_size(const size_t step_bytes);

    /**
       Get the minimum memory chunk size
       \ingroup device_func_mem
    */
    AFAPI af_err af_get_mem_step_size(size_t *step_bytes);

#if AF_API_VERSION >= 31
    /**
       Lock the device buffer in the memory manager.

       Locked buffers are not freed by memory manager until \ref af_unlock_array is called.
       \ingroup device_func_mem
    */
#if AF_API_VERSION >= 33
    AF_DEPRECATED("Use af_lock_array instead")
#endif
    AFAPI af_err af_lock_device_ptr(const af_array arr);
#endif

#if AF_API_VERSION >= 31
    /**
       Unlock device buffer in the memory manager.

       This function will give back the control over the device pointer to the memory manager.
       \ingroup device_func_mem
    */
#if AF_API_VERSION >= 33
    AF_DEPRECATED("Use af_unlock_array instead")
#endif
    AFAPI af_err af_unlock_device_ptr(const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       Lock the device buffer in the memory manager.

       Locked buffers are not freed by memory manager until \ref af_unlock_array is called.
       \ingroup device_func_mem
    */
    AFAPI af_err af_lock_array(const af_array arr);
#endif

#if AF_API_VERSION >= 33
    /**
       Unlock device buffer in the memory manager.

       This function will give back the control over the device pointer to the memory manager.
       \ingroup device_func_mem
    */
    AFAPI af_err af_unlock_array(const af_array arr);
#endif

#if AF_API_VERSION >= 34
    /**
       Query if the array has been locked by the user.

       An array can be locked by the user by calling `af_lock_array`
       or `af_get_device_ptr` or `af_get_raw_ptr` function.

       \ingroup device_func_mem
    */
    AFAPI af_err af_is_locked_array(bool *res, const af_array arr);
#endif

    /**
       Get the device pointer and lock the buffer in memory manager.

       The device pointer \p ptr is notfreed by memory manager until \ref af_unlock_device_ptr is called.
       \ingroup device_func_mem

       \note For OpenCL backend *ptr should be cast to cl_mem.
    */
    AFAPI af_err af_get_device_ptr(void **ptr, const af_array arr);


#ifdef __cplusplus
}
#endif
