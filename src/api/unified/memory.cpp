/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/memory.h>
#include "symbol_manager.hpp"

af_err af_create_memory_manager(af_memory_manager* out) {
    CALL(af_create_memory_manager, out);
}

af_err af_release_memory_manager(af_memory_manager handle) {
    CALL(af_release_memory_manager, handle);
}

af_err af_set_memory_manager(af_memory_manager handle) {
    CALL(af_set_memory_manager, handle);
}

af_err af_set_memory_manager_pinned(af_memory_manager handle) {
    CALL(af_set_memory_manager_pinned, handle);
}

af_err af_unset_memory_manager() { CALL_NO_PARAMS(af_unset_memory_manager); }

af_err af_unset_memory_manager_pinned() {
    CALL_NO_PARAMS(af_unset_memory_manager_pinned);
}

af_err af_memory_manager_get_payload(af_memory_manager handle, void** payload) {
    CALL(af_memory_manager_get_payload, handle, payload);
}

af_err af_memory_manager_set_payload(af_memory_manager handle, void* payload) {
    CALL(af_memory_manager_set_payload, handle, payload);
}

af_err af_memory_manager_set_initialize_fn(af_memory_manager handle,
                                           af_memory_manager_initialize_fn fn) {
    CALL(af_memory_manager_set_initialize_fn, handle, fn);
}

af_err af_memory_manager_set_shutdown_fn(af_memory_manager handle,
                                         af_memory_manager_shutdown_fn fn) {
    CALL(af_memory_manager_set_shutdown_fn, handle, fn);
}

af_err af_memory_manager_set_alloc_fn(af_memory_manager handle,
                                      af_memory_manager_alloc_fn fn) {
    CALL(af_memory_manager_set_alloc_fn, handle, fn);
}

af_err af_memory_manager_set_allocated_fn(af_memory_manager handle,
                                          af_memory_manager_allocated_fn fn) {
    CALL(af_memory_manager_set_allocated_fn, handle, fn);
}

af_err af_memory_manager_set_unlock_fn(af_memory_manager handle,
                                       af_memory_manager_unlock_fn fn) {
    CALL(af_memory_manager_set_unlock_fn, handle, fn);
}

af_err af_memory_manager_set_signal_memory_cleanup_fn(
    af_memory_manager handle, af_memory_manager_signal_memory_cleanup_fn fn) {
    CALL(af_memory_manager_set_signal_memory_cleanup_fn, handle, fn);
}

af_err af_memory_manager_set_print_info_fn(af_memory_manager handle,
                                           af_memory_manager_print_info_fn fn) {
    CALL(af_memory_manager_set_print_info_fn, handle, fn);
}

af_err af_memory_manager_set_user_lock_fn(af_memory_manager handle,
                                          af_memory_manager_user_lock_fn fn) {
    CALL(af_memory_manager_set_user_lock_fn, handle, fn);
}

af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn) {
    CALL(af_memory_manager_set_user_unlock_fn, handle, fn);
}

af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn) {
    CALL(af_memory_manager_set_is_user_locked_fn, handle, fn);
}

af_err af_memory_manager_set_get_memory_pressure_fn(
    af_memory_manager handle, af_memory_manager_get_memory_pressure_fn fn) {
    CALL(af_memory_manager_set_get_memory_pressure_fn, handle, fn);
}

af_err af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    af_memory_manager handle,
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn fn) {
    CALL(af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn, handle, fn);
}

af_err af_memory_manager_set_add_memory_management_fn(
    af_memory_manager handle, af_memory_manager_add_memory_management_fn fn) {
    CALL(af_memory_manager_set_add_memory_management_fn, handle, fn);
}

af_err af_memory_manager_set_remove_memory_management_fn(
    af_memory_manager handle,
    af_memory_manager_remove_memory_management_fn fn) {
    CALL(af_memory_manager_set_remove_memory_management_fn, handle, fn);
}

af_err af_memory_manager_get_active_device_id(af_memory_manager handle,
                                              int* id) {
    CALL(af_memory_manager_get_active_device_id, handle, id);
}

af_err af_memory_manager_native_alloc(af_memory_manager handle, void** ptr,
                                      size_t size) {
    CALL(af_memory_manager_native_alloc, handle, ptr, size);
}

af_err af_memory_manager_native_free(af_memory_manager handle, void* ptr) {
    CALL(af_memory_manager_native_free, handle, ptr);
}

af_err af_memory_manager_get_max_memory_size(af_memory_manager handle,
                                             size_t* size, int id) {
    CALL(af_memory_manager_get_max_memory_size, handle, size, id);
}

af_err af_memory_manager_get_memory_pressure_threshold(af_memory_manager handle,
                                                       float* value) {
    CALL(af_memory_manager_get_memory_pressure_threshold, handle, value);
}

af_err af_memory_manager_set_memory_pressure_threshold(af_memory_manager handle,
                                                       float value) {
    CALL(af_memory_manager_set_memory_pressure_threshold, handle, value);
}
