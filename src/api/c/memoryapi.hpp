/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <memory_manager.hpp>

#include <events.hpp>
#include <af/event.h>
#include <af/memory.h>

////////////////////////////////////////////////////////////////////////////////
// Buffer Info
////////////////////////////////////////////////////////////////////////////////

struct BufferInfo {
    void *ptr;
    af_event event;
};

BufferInfo &getBufferInfo(const af_buffer_info pair);

af_buffer_info getHandle(BufferInfo &pairHandle);

detail::Event &getEventFromBufferInfoHandle(const af_buffer_info handle);

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

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
    af_memory_manager_add_memory_management add_memory_management_fn;
    af_memory_manager_remove_memory_management remove_memory_management_fn;
    af_memory_manager_jit_tree_exceeds_memory_pressure_fn
        jit_tree_exceeds_memory_pressure_fn;
    // A generic payload on which data can be stored on the af_memory_manager
    // and is accessible from the handle
    void *payload;
};

MemoryManager &getMemoryManager(const af_memory_manager manager);

af_memory_manager getHandle(MemoryManager &manager);

/**
 * An internal wrapper around an af_memory_manager which calls function pointers
 * on a af_memory_manager via calls to a MemoryManagerBase
 */
class MemoryManagerFunctionWrapper : public common::memory::MemoryManagerBase {
    af_memory_manager handle_;

   public:
    MemoryManagerFunctionWrapper(af_memory_manager handle) : handle_(handle) {}
    ~MemoryManagerFunctionWrapper() {}
    void initialize() override;
    void shutdown() override;
    af_buffer_info alloc(const size_t size, bool user_lock) override;
    size_t allocated(void *ptr) override;
    void unlock(void *ptr, af_event e, bool user_unlock) override;
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
