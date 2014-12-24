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
    AFAPI void info();

    AFAPI void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    AFAPI int getDeviceCount();

    AFAPI int getDevice();

    AFAPI bool isDoubleAvailable(const int device);

    AFAPI void setDevice(const int device);

    AFAPI void sync(int device = -1);

    AFAPI void *alloc(size_t elements, dtype type);

    template<typename T>
    T* alloc(size_t elements);

    AFAPI void *pinned(size_t elements, dtype type);

    template<typename T>
    T* pinned(size_t elements);

    AFAPI void free(const void *);

    AFAPI void freePinned(const void *);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_info();

    AFAPI af_err af_init();

    AFAPI af_err af_deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    AFAPI af_err af_get_device_count(int *num_of_devices);

    AFAPI af_err af_get_dbl_support(bool* available, const int device);

    AFAPI af_err af_set_device(const int device);

    AFAPI af_err af_get_device(int *device);

    AFAPI af_err af_sync(const int device);

    AFAPI af_err af_device_array(af_array *arr, const void *data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_get_device_ptr(void **ptr, const af_array arr, bool read_only);

    AFAPI af_err af_alloc_device(void **ptr, dim_type bytes);

    AFAPI af_err af_alloc_pinned(void **ptr, dim_type bytes);

    AFAPI af_err af_free_device(void *ptr);

    AFAPI af_err af_free_pinned(void *ptr);

#ifdef __cplusplus
}
#endif
