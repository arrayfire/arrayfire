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
#if defined(WITH_GRAPHICS)
#include <forge.h>
#endif

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
       \defgroup device_func_prop deviceprop

       Get device properties

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);
    /**
       @}
    */

    /**
       \defgroup device_func_count getDeviceCount

       Get the number of devices available for the current backend

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI int getDeviceCount();
    /**
       @}
    */

    /**
       \defgroup device_func_get getDevice

       Get the current device ID

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI int getDevice();
    /**
       @}
    */

    /**
       \defgroup device_func_dbl isDoubleAvailable

       Check if double precision support is available for specified device

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI bool isDoubleAvailable(const int device);
    /**
       @}
    */

    /**
       \defgroup device_func_set setDevice

       Change current device to specified device

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void setDevice(const int device);
    /**
       @}
    */

    /**
       \defgroup device_func_sync sync

       Wait until all operations on device are finished

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void sync(int device = -1);
    /**
       @}
    */

    /**
      \defgroup device_func_alloc alloc

       Allocate device memory using ArrayFire's memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void *alloc(size_t elements, dtype type);

    template<typename T>
    T* alloc(size_t elements);
    /**
       @}
    */


    /**
       \defgroup device_func_pinned pinned

       Allocate pinned memory using ArrayFire's memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void *pinned(size_t elements, dtype type);

    template<typename T>
    T* pinned(size_t elements);
    /**
       @}
    */

    /**
       \defgroup device_func_free free

       Free device memory allocated by ArrayFire's memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void free(const void *);
    /**
       @}
    */

    /**
       \defgroup device_func_free_pinned freePinned

       Free pinned memory allocated by ArrayFire' memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */
    AFAPI void freePinned(const void *);

    /**
       @}
    */

    /**
       \defgroup device_func_mem deviceMemInfo

       Get the information of memory used by the memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */

    AFAPI void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                             size_t *lock_bytes, size_t *lock_buffers);

    /**
       \defgroup device_func_gc deviceGC

       Call the garbage collection function in the memory manager

       @{

       \ingroup arrayfire_func
       \ingroup device_mat
    */

    AFAPI void deviceGC();

    AFAPI void initGraphics(int device);
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
       \ingroup device_func_prop
    */
    AFAPI af_err af_deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

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
    AFAPI af_err af_get_device_ptr(void **ptr, const af_array arr, bool read_only);

    /**
       \ingroup device_func_alloc
    */
    AFAPI af_err af_alloc_device(void **ptr, dim_type bytes);

    /**
       \ingroup device_func_pinned
    */
    AFAPI af_err af_alloc_pinned(void **ptr, dim_type bytes);

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
    AFAPI af_err af_device_array(af_array *arr, const void *data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    /**
       Get memory information from the memory manager
       \ingroup device_func_mem
    */
    AFAPI af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                                    size_t *lock_bytes, size_t *lock_buffers);

    /**
       Call the garbage collection routine
       \ingroup device_func_gc
    */
    AFAPI af_err af_device_gc();

    AFAPI af_err af_init_graphics(int device);

#ifdef __cplusplus
}
#endif
