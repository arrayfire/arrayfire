/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/MemoryManagerBase.hpp>

#include <af/memory.h>

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

/**
 * An internal wrapper around an af_memory_manager which calls function pointers
 * on a af_memory_manager via calls to a MemoryManagerBase
 */
class MemoryManagerFunctionWrapper final
    : public arrayfire::common::MemoryManagerBase {
    af_memory_manager handle_;

   public:
    MemoryManagerFunctionWrapper(af_memory_manager handle);
    ~MemoryManagerFunctionWrapper();
    void initialize() override;
    void shutdown() override;
    void *alloc(bool user_lock, const unsigned ndims, dim_t *dims,
                const unsigned element_size) override;
    size_t allocated(void *ptr) override;
    void unlock(void *ptr, bool user_unlock) override;
    void signalMemoryCleanup() override;
    void printInfo(const char *msg, const int device) override;
    void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                   size_t *lock_bytes, size_t *lock_buffers) override;
    void userLock(const void *ptr) override;
    void userUnlock(const void *ptr) override;
    bool isUserLocked(const void *ptr) override;
    size_t getMemStepSize() override;
    void setMemStepSize(size_t new_step_size) override;
    float getMemoryPressure() override;
    bool jitTreeExceedsMemoryPressure(size_t bytes) override;

    void addMemoryManagement(int device) override;
    void removeMemoryManagement(int device) override;
};

struct MemoryManager {
    // Callbacks from public API
    af_memory_manager_initialize_fn initialize_fn;
    af_memory_manager_shutdown_fn shutdown_fn;
    af_memory_manager_alloc_fn alloc_fn;
    af_memory_manager_allocated_fn allocated_fn;
    af_memory_manager_unlock_fn unlock_fn;
    af_memory_manager_print_info_fn print_info_fn;
    af_memory_manager_user_lock_fn user_lock_fn;
    af_memory_manager_user_unlock_fn user_unlock_fn;
    af_memory_manager_is_user_locked_fn is_user_locked_fn;
    af_memory_manager_get_memory_pressure_fn get_memory_pressure_fn;
    af_memory_manager_signal_memory_cleanup_fn signal_memory_cleanup_fn;
    af_memory_manager_add_memory_management_fn add_memory_management_fn;
    af_memory_manager_remove_memory_management_fn remove_memory_management_fn;
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn
        jit_tree_exceeds_memory_pressure_fn;
    // A generic payload on which data can be stored on the af_memory_manager
    // and is accessible from the handle
    void *payload;
    // A pointer to the MemoryManagerFunctionWrapper wrapping this struct that
    // facilitates calling native memory functions directly from the handle. The
    // lifetime of the wrapper is controlled by the relevant device manager
    MemoryManagerFunctionWrapper *wrapper;
};

MemoryManager &getMemoryManager(const af_memory_manager handle);

af_memory_manager getHandle(MemoryManager &manager);
