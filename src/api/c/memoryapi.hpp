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
    af_memory_manager_garbage_collect_fn garbage_collect_fn;
    af_memory_manager_print_info_fn print_info_fn;
    af_memory_manager_usage_info_fn usage_info_fn;
    af_memory_manager_user_lock_fn user_lock_fn;
    af_memory_manager_user_unlock_fn user_unlock_fn;
    af_memory_manager_is_user_locked_fn is_user_locked_fn;
    af_memory_manager_get_mem_step_size_fn get_mem_step_size_fn;
    af_memory_manager_get_max_bytes_fn get_max_bytes_fn;
    af_memory_manager_get_max_buffers_fn get_max_buffers_fn;
    af_memory_manager_set_mem_step_size_fn set_mem_step_size_fn;
    af_memory_manager_check_memory_limit check_memory_limit_fn;
    af_memory_manager_add_memory_management add_memory_management_fn;
    af_memory_manager_remove_memory_management remove_memory_management_fn;
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

    void initialize() override;
    void shutdown() override;
    af_buffer_info alloc(const size_t size, bool user_lock);
    size_t allocated(void *ptr);
    void unlock(void *ptr, af_event e, bool user_unlock);
    void garbageCollect();
    void printInfo(const char *msg, const int device);
    void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                   size_t *lock_bytes, size_t *lock_buffers);
    void userLock(const void *ptr);
    void userUnlock(const void *ptr);
    bool isUserLocked(const void *ptr);
    size_t getMemStepSize();
    size_t getMaxBytes();
    unsigned getMaxBuffers();
    void setMemStepSize(size_t new_step_size);
    bool checkMemoryLimit();

    void addMemoryManagement(int device);
    void removeMemoryManagement(int device);
};
