/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include "ArrayFireDefaultMemoryManager.hpp"

#include <cstdio>

ArrayFireDefaultMemoryManager payload(1000, false);

af_err alloc_fn(af_memory_manager manager, void **ptr, int userLock,
                const unsigned ndims, dim_t *dims,
                const unsigned element_size) {
    printf("MM: allocate\n");
    *ptr = payload.alloc(userLock, ndims, dims, element_size);
    return AF_SUCCESS;
}

af_err allocated_fn(af_memory_manager manager, size_t *out, void *ptr) {
    payload.allocated(ptr);
    return AF_SUCCESS;
}

af_err unlock_fn(af_memory_manager manager, void *ptr, int userLock) {
    printf("MM: unlock\n");
    payload.unlock(ptr, userLock);
    return AF_SUCCESS;
}

af_err signal_memory_cleanup_fn(af_memory_manager manager) {
    printf("MM: cleanup\n");
    payload.signalMemoryCleanup();
    return AF_SUCCESS;
}

af_err user_lock_fn(af_memory_manager manager, void *ptr) {
    printf("MM: lock\n");
    payload.userLock(ptr);
    return AF_SUCCESS;
}

af_err user_unlock_fn(af_memory_manager manager, void *ptr) {
    printf("MM: unlock\n");
    payload.userUnlock(ptr);
    return AF_SUCCESS;
}

af_err is_user_locked_fn(af_memory_manager manager, int *out, void *ptr) {
    printf("MM: user unlock\n");
    *out = payload.isUserLocked(ptr);
    return AF_SUCCESS;
}

af_err get_memory_pressure_fn(af_memory_manager manager, float *pressure) {
    *pressure = payload.getMemoryPressure();
    printf("MM: Getting memory pressure: %f\n", *pressure);
    return AF_SUCCESS;
}

af_err jit_tree_exceeds_memory_pressure_fn(af_memory_manager manager, int *out,
                                           size_t size) {
    *out = payload.jitTreeExceedsMemoryPressure(size);
    printf("MM: JIT exceeds memory pressure: %d\n", *out);
    return AF_SUCCESS;
}

af_err initialize_fn(af_memory_manager manager) {
    printf("MM: initialize\n");
    payload.initialize();
    return AF_SUCCESS;
};

af_err shutdown_fn(af_memory_manager manager) {
    printf("MM: shutdown\n");
    payload.shutdown();
    return AF_SUCCESS;
};

af_err print_info_fn(af_memory_manager manager, char *msg, int device) {
    printf("MM: printing memory informaion\n");
    payload.printInfo(msg, device);
    return AF_SUCCESS;
}

int main(int argc, char *argv[]) {
    printf(
        "ArrayFire custom memory manager example.\n\n"
        "This example shows how you can customize the ArrayFire memory\n"
        "allocation functionallity. This example replaces the default\n"
        "memory manager with a custom memory manager that mimics the\n"
        "default memory manager's behavior.\n\n");

    af_memory_manager manager = nullptr;
    af_create_memory_manager(&manager);

    // Set payload_fn
    af_memory_manager_set_payload(manager, &payload);

    af_memory_manager_set_initialize_fn(manager, initialize_fn);
    af_memory_manager_set_shutdown_fn(manager, shutdown_fn);

    // alloc
    af_memory_manager_set_alloc_fn(manager, alloc_fn);
    af_memory_manager_set_allocated_fn(manager, allocated_fn);
    af_memory_manager_set_unlock_fn(manager, unlock_fn);
    // utils
    af_memory_manager_set_signal_memory_cleanup_fn(manager,
                                                   signal_memory_cleanup_fn);
    af_memory_manager_set_print_info_fn(manager, print_info_fn);

    // user lock/unlock
    af_memory_manager_set_user_lock_fn(manager, user_lock_fn);
    af_memory_manager_set_user_unlock_fn(manager, user_unlock_fn);
    af_memory_manager_set_is_user_locked_fn(manager, is_user_locked_fn);

    // memory pressure
    af_memory_manager_set_get_memory_pressure_fn(manager,
                                                 get_memory_pressure_fn);
    af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
        manager, jit_tree_exceeds_memory_pressure_fn);

    af_set_memory_manager(manager);

    af::info();
    printf("\nCreating new array object.\n");
    af::array test_array = af::randu(10, 10);
    af_print(test_array);

    af::array a = constant(5, af::dim4(10, 10));
    af::array c = a * a;
    printf("Performing JIT operations.\n");
    c.eval();

    printf("Printing JIT'd array\n");
    af_print(c);

    printf("Getting device function.\n");
    void *ptr = test_array.device<float>();

    printf("Unlocking.\n");
    test_array.unlock();

    af::printMemInfo();

    printf("Calling device garbage collector.\n");
    af::deviceGC();
    af::sync();

    return 0;
}
