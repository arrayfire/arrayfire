/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/deprecated.hpp>
#include <af/array.h>
#include <af/backend.h>
#include <af/device.h>
#include "symbol_manager.hpp"

af_err af_set_backend(const af_backend bknd) {
    return arrayfire::unified::setBackend(bknd);
}

af_err af_get_backend_count(unsigned *num_backends) {
    *num_backends =
        arrayfire::unified::AFSymbolManager::getInstance().getBackendCount();
    return AF_SUCCESS;
}

af_err af_get_available_backends(int *result) {
    *result = arrayfire::unified::AFSymbolManager::getInstance()
                  .getAvailableBackends();
    return AF_SUCCESS;
}

af_err af_get_backend_id(af_backend *result, const af_array in) {
    // DO NOT CALL CHECK_ARRAYS HERE.
    // IT WILL RESULT IN AN INFINITE RECURSION
    CALL(af_get_backend_id, result, in);
}

af_err af_get_device_id(int *device, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_get_device_id, device, in);
}

af_err af_get_active_backend(af_backend *result) {
    *result = arrayfire::unified::getActiveBackend();
    return AF_SUCCESS;
}

af_err af_info() { CALL_NO_PARAMS(af_info); }

af_err af_init() { CALL_NO_PARAMS(af_init); }

af_err af_info_string(char **str, const bool verbose) {
    CALL(af_info_string, str, verbose);
}

af_err af_device_info(char *d_name, char *d_platform, char *d_toolkit,
                      char *d_compute) {
    CALL(af_device_info, d_name, d_platform, d_toolkit, d_compute);
}

af_err af_get_device_count(int *num_of_devices) {
    CALL(af_get_device_count, num_of_devices);
}

af_err af_get_dbl_support(bool *available, const int device) {
    CALL(af_get_dbl_support, available, device);
}

af_err af_get_half_support(bool *available, const int device) {
    CALL(af_get_half_support, available, device);
}

af_err af_set_device(const int device) { CALL(af_set_device, device); }

af_err af_get_device(int *device) { CALL(af_get_device, device); }

af_err af_sync(const int device) { CALL(af_sync, device); }

af_err af_alloc_device(void **ptr, const dim_t bytes) {
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_alloc_device, ptr, bytes);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_alloc_device_v2(void **ptr, const dim_t bytes) {
    CALL(af_alloc_device_v2, ptr, bytes);
}

af_err af_alloc_pinned(void **ptr, const dim_t bytes) {
    CALL(af_alloc_pinned, ptr, bytes);
}

af_err af_free_device(void *ptr) {
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_free_device, ptr);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_free_device_v2(void *ptr) { CALL(af_free_device_v2, ptr); }

af_err af_free_pinned(void *ptr) { CALL(af_free_pinned, ptr); }

af_err af_alloc_host(void **ptr, const dim_t bytes) {
    *ptr = malloc(bytes);  // NOLINT(hicpp-no-malloc)
    return (*ptr == NULL) ? AF_ERR_NO_MEM : AF_SUCCESS;
}

af_err af_free_host(void *ptr) {
    free(ptr);  // NOLINT(hicpp-no-malloc)
    return AF_SUCCESS;
}

af_err af_device_array(af_array *arr, void *data, const unsigned ndims,
                       const dim_t *const dims, const af_dtype type) {
    CALL(af_device_array, arr, data, ndims, dims, type);
}

af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes, size_t *lock_buffers) {
    CALL(af_device_mem_info, alloc_bytes, alloc_buffers, lock_bytes,
         lock_buffers);
}

af_err af_print_mem_info(const char *msg, const int device_id) {
    CALL(af_print_mem_info, msg, device_id);
}

af_err af_device_gc() { CALL_NO_PARAMS(af_device_gc); }

af_err af_set_mem_step_size(const size_t step_bytes) {
    CALL(af_set_mem_step_size, step_bytes);
}

af_err af_get_mem_step_size(size_t *step_bytes) {
    CALL(af_get_mem_step_size, step_bytes);
}

af_err af_lock_device_ptr(const af_array arr) {
    CHECK_ARRAYS(arr);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_lock_device_ptr, arr);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_unlock_device_ptr(const af_array arr) {
    CHECK_ARRAYS(arr);
    AF_DEPRECATED_WARNINGS_OFF
    CALL(af_unlock_device_ptr, arr);
    AF_DEPRECATED_WARNINGS_ON
}

af_err af_lock_array(const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_lock_array, arr);
}

af_err af_unlock_array(const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_unlock_array, arr);
}

af_err af_is_locked_array(bool *res, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_is_locked_array, res, arr);
}

af_err af_get_device_ptr(void **ptr, const af_array arr) {
    CHECK_ARRAYS(arr);
    CALL(af_get_device_ptr, ptr, arr);
}

af_err af_eval_multiple(const int num, af_array *arrays) {
    for (int i = 0; i < num; i++) { CHECK_ARRAYS(arrays[i]); }
    CALL(af_eval_multiple, num, arrays);
}

af_err af_set_manual_eval_flag(bool flag) {
    CALL(af_set_manual_eval_flag, flag);
}

af_err af_get_manual_eval_flag(bool *flag) {
    CALL(af_get_manual_eval_flag, flag);
}

af_err af_set_kernel_cache_directory(const char *path, int override_eval) {
    CALL(af_set_kernel_cache_directory, path, override_eval);
}

af_err af_get_kernel_cache_directory(size_t *length, char *path) {
    CALL(af_get_kernel_cache_directory, length, path);
}
