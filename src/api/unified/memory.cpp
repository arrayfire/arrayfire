/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/memory.h>
#include "symbol_manager.hpp"

af_err af_create_memory_manager(af_memory_manager* out) { return CALL(out); }

af_err af_release_memory_manager(af_memory_manager handle) {
    return CALL(handle);
}

af_err af_set_memory_manager(af_memory_manager handle) { return CALL(handle); }

af_err af_set_memory_manager_pinned(af_memory_manager handle) {
    return CALL(handle);
}

af_err af_unset_memory_manager() { return CALL_NO_PARAMS(); }

af_err af_unset_memory_manager_pinned() { return CALL_NO_PARAMS(); }

af_err af_memory_manager_get_payload(af_memory_manager handle, void** payload) {
    return CALL(handle, payload);
}

af_err af_memory_manager_set_payload(af_memory_manager handle, void* payload) {
    return CALL(handle, payload);
}

af_err af_memory_manager_set_initialize_fn(af_memory_manager handle,
                                           af_memory_manager_initialize_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_shutdown_fn(af_memory_manager handle,
                                         af_memory_manager_shutdown_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_alloc_fn(af_memory_manager handle,
                                      af_memory_manager_alloc_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_allocated_fn(af_memory_manager handle,
                                          af_memory_manager_allocated_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_unlock_fn(af_memory_manager handle,
                                       af_memory_manager_unlock_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_signal_memory_cleanup_fn(
    af_memory_manager handle, af_memory_manager_signal_memory_cleanup_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_print_info_fn(af_memory_manager handle,
                                           af_memory_manager_print_info_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_user_lock_fn(af_memory_manager handle,
                                          af_memory_manager_user_lock_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_get_memory_pressure_fn(
    af_memory_manager handle, af_memory_manager_get_memory_pressure_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    af_memory_manager handle,
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_add_memory_management_fn(
    af_memory_manager handle, af_memory_manager_add_memory_management_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_set_remove_memory_management_fn(
    af_memory_manager handle, af_memory_manager_remove_memory_management_fn fn) {
    return CALL(handle, fn);
}

af_err af_memory_manager_get_active_device_id(af_memory_manager handle,
                                              int* id) {
    return CALL(handle, id);
}

af_err af_memory_manager_native_alloc(af_memory_manager handle, void** ptr,
                                      size_t size) {
    return CALL(handle, ptr, size);
}

af_err af_memory_manager_native_free(af_memory_manager handle, void* ptr) {
    return CALL(handle, ptr);
}

af_err af_memory_manager_get_max_memory_size(af_memory_manager handle,
                                             size_t* size, int id) {
    return CALL(handle, size, id);
}

af_err af_memory_manager_get_memory_pressure_threshold(af_memory_manager handle,
                                                       float* value) {
    return CALL(handle, value);
}

af_err af_memory_manager_set_memory_pressure_threshold(af_memory_manager handle,
                                                       float value) {
    return CALL(handle, value);
}
