/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/backend.h>
#include <af/device.h>
#include "symbol_manager.hpp"

af_err af_set_backend(const af_backend bknd) {
    return unified::AFSymbolManager::getInstance().setBackend(bknd);
}

af_err af_get_backend_count(unsigned *num_backends) {
    *num_backends = unified::AFSymbolManager::getInstance().getBackendCount();
    return AF_SUCCESS;
}

af_err af_get_available_backends(int *result) {
    *result = unified::AFSymbolManager::getInstance().getAvailableBackends();
    return AF_SUCCESS;
}

af_err af_get_backend_id(af_backend *result, const af_array in) {
    // DO NOT CALL CHECK_ARRAYS HERE.
    // IT WILL RESULT IN AN INFINITE RECURSION
    return CALL(result, in);
}

af_err af_get_device_id(int *device, const af_array in) {
    CHECK_ARRAYS(in);
    return CALL(device, in);
}

af_err af_get_active_backend(af_backend *result) {
    *result = unified::AFSymbolManager::getInstance().getActiveBackend();
    return AF_SUCCESS;
}

af_err af_info() { return CALL_NO_PARAMS(); }

af_err af_init() { return CALL_NO_PARAMS(); }

af_err af_info_string(char **str, const bool verbose) {
    return CALL(str, verbose);
}

af_err af_device_info(char *d_name, char *d_platform, char *d_toolkit,
                      char *d_compute) {
    return CALL(d_name, d_platform, d_toolkit, d_compute);
}

af_err af_get_device_count(int *num_of_devices) { return CALL(num_of_devices); }

af_err af_get_dbl_support(bool *available, const int device) {
    return CALL(available, device);
}

af_err af_set_device(const int device) { return CALL(device); }

af_err af_get_device(int *device) { return CALL(device); }

af_err af_sync(const int device) { return CALL(device); }

af_err af_alloc_device(void **ptr, const dim_t bytes) {
    return CALL(ptr, bytes);
}

af_err af_alloc_pinned(void **ptr, const dim_t bytes) {
    return CALL(ptr, bytes);
}

af_err af_free_device(void *ptr) { return CALL(ptr); }

af_err af_free_pinned(void *ptr) { return CALL(ptr); }

af_err af_alloc_host(void **ptr, const dim_t bytes) {
    *ptr = malloc(bytes);
    return (*ptr == NULL) ? AF_ERR_NO_MEM : AF_SUCCESS;
}

af_err af_free_host(void *ptr) {
    free(ptr);
    return AF_SUCCESS;
}

af_err af_device_array(af_array *arr, void *data, const unsigned ndims,
                       const dim_t *const dims, const af_dtype type) {
    return CALL(arr, data, ndims, dims, type);
}

af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes, size_t *lock_buffers) {
    return CALL(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers);
}

af_err af_print_mem_info(const char *msg, const int device_id) {
    return CALL(msg, device_id);
}

af_err af_device_gc() { return CALL_NO_PARAMS(); }

af_err af_set_mem_step_size(const size_t step_bytes) {
    return CALL(step_bytes);
}

af_err af_get_mem_step_size(size_t *step_bytes) { return CALL(step_bytes); }

af_err af_lock_device_ptr(const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(arr);
}

af_err af_unlock_device_ptr(const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(arr);
}

af_err af_lock_array(const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(arr);
}

af_err af_unlock_array(const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(arr);
}

af_err af_is_locked_array(bool *res, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(res, arr);
}

af_err af_get_device_ptr(void **ptr, const af_array arr) {
    CHECK_ARRAYS(arr);
    return CALL(ptr, arr);
}

af_err af_eval_multiple(const int num, af_array *arrays) {
    for (int i = 0; i < num; i++) { CHECK_ARRAYS(arrays[i]); }
    return CALL(num, arrays);
}

af_err af_set_manual_eval_flag(bool flag) { return CALL(flag); }

af_err af_get_manual_eval_flag(bool *flag) { return CALL(flag); }
