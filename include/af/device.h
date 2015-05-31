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
       \defgroup device_func_prop deviceInfo

       Get device information

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void deviceInfo(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);
    /**
       @}
    */

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
    /// \returns true if the \p device supports double precision operations. false otherwise
    /// \ingroup device_func_dbl
    AFAPI bool isDoubleAvailable(const int device);

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
    AFAPI void *alloc(const size_t elements, const dtype type);

    /// \brief Allocates memory using ArrayFire's memory manager
    //
    /// \copydoc device_func_alloc
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \note the size of the memory allocated is the number of \p elements *
    ///         sizeof(type)
    template<typename T>
    T* alloc(const size_t elements);
    /// @}

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

    /// \ingroup device_func_free
    /// @{
    /// \copydoc device_func_free
    /// \param[in] ptr the memory to free
    AFAPI void free(const void *ptr);

    /// \copydoc free()
    AFAPI void freePinned(const void *ptr);
    ///@}

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
    AFAPI void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                             size_t *lock_bytes, size_t *lock_buffers);

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

    AFAPI af_err af_init();

    /**
       \ingroup device_func_info
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
       \ingroup device_func_device
    */
    AFAPI af_err af_get_device_ptr(void **ptr, const af_array arr);

    /**
       \ingroup device_func_alloc
    */
    AFAPI af_err af_alloc_device(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_pinned
    */
    AFAPI af_err af_alloc_pinned(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_free
    */
    AFAPI af_err af_free_device(void *ptr);

    /**
       \ingroup device_func_free_pinned
    */
    AFAPI af_err af_free_pinned(void *ptr);

    /**
       Create array from device memory
       \ingroup construct_mat
    */
    AFAPI af_err af_device_array(af_array *arr, const void *data, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       Get memory information from the memory manager
       \ingroup device_func_mem
    */
    AFAPI af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                                    size_t *lock_bytes, size_t *lock_buffers);

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

#ifdef __cplusplus
}
#endif
