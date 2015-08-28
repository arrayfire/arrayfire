/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/hapi.h>
#include <af/device.h>
#include "symbol_manager.hpp"

af_err af_set_backend(const af_backend bknd)
{
    return AFSymbolManager::getInstance().setBackend(bknd);
}

af_err af_info()
{
    return CALL_NO_PARAMS();
}

AFAPI af_err af_init()
{
    return CALL_NO_PARAMS();
}

AFAPI af_err af_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    return CALL(d_name, d_platform, d_toolkit, d_compute);
}

AFAPI af_err af_get_device_count(int *num_of_devices)
{
    return CALL(num_of_devices);
}

AFAPI af_err af_get_dbl_support(bool* available, const int device)
{
    return CALL(available, device);
}

AFAPI af_err af_set_device(const int device)
{
    return CALL(device);
}

AFAPI af_err af_get_device(int *device)
{
    return CALL(device);
}

AFAPI af_err af_sync(const int device)
{
    return CALL(device);
}

AFAPI af_err af_alloc_device(void **ptr, const dim_t bytes)
{
    return CALL(ptr, bytes);
}

AFAPI af_err af_alloc_pinned(void **ptr, const dim_t bytes)
{
    return CALL(ptr, bytes);
}

AFAPI af_err af_free_device(void *ptr)
{
    return CALL(ptr);
}

AFAPI af_err af_free_pinned(void *ptr)
{
    return CALL(ptr);
}

AFAPI af_err af_device_array(af_array *arr, const void *data, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(arr, data, ndims, dims, type);
}

AFAPI af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
        size_t *lock_bytes, size_t *lock_buffers)
{
    return CALL(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers);
}

AFAPI af_err af_device_gc()
{
    return CALL_NO_PARAMS();
}

AFAPI af_err af_set_mem_step_size(const size_t step_bytes)
{
    return CALL(step_bytes);
}

AFAPI af_err af_get_mem_step_size(size_t *step_bytes)
{
    return CALL(step_bytes);
}

AFAPI af_err af_lock_device_ptr(const af_array arr)
{
    return CALL(arr);
}

AFAPI af_err af_unlock_device_ptr(const af_array arr)
{
    return CALL(arr);
}

AFAPI af_err af_get_device_ptr(void **ptr, const af_array arr)
{
    return CALL(ptr, arr);
}
