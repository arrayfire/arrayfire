/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <af/event.h>
#include <af/memory.h>

namespace af {

buffer_info::buffer_info(void* ptr, af_event event) {
    AF_CHECK(af_create_buffer_info(&p_, ptr, event));
}

buffer_info::buffer_info(af_buffer_info p) : p_(p) {}

buffer_info::buffer_info(buffer_info&& other) : p_(other.p_) { other.p_ = 0; }

buffer_info& buffer_info::operator=(buffer_info&& other) {
    af_delete_buffer_info(this->p_);
    this->p_ = other.p_;
    other.p_ = 0;
    return *this;
}

buffer_info::~buffer_info() {
    // No throw dtor
    af_delete_buffer_info(p_);
}

void* buffer_info::getPtr() const {
    void* ptr;
    AF_CHECK(af_buffer_info_get_ptr(&ptr, p_));
    return ptr;
}

af_event buffer_info::getEvent() const {
    af_event e;
    AF_CHECK(af_buffer_info_get_event(&e, p_));
    return e;
}

void buffer_info::setPtr(void* ptr) {
    AF_CHECK(af_buffer_info_set_ptr(p_, ptr));
}

void buffer_info::setEvent(af_event event) {
    AF_CHECK(af_buffer_info_set_event(p_, event));
}

af_event buffer_info::unlockEvent() {
    af_event event;
    AF_CHECK(af_unlock_buffer_info_event(&event, p_));
    // Zero out the event
    AF_CHECK(af_buffer_info_set_event(p_, 0));
    return event;
}

void* buffer_info::unlockPtr() {
    void* ptr;
    AF_CHECK(af_unlock_buffer_info_ptr(&ptr, p_));
    // Zero out the ptr
    AF_CHECK(af_buffer_info_set_ptr(p_, 0));
    return ptr;
}

af_buffer_info buffer_info::get() const { return p_; }

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

memory_manager::memory_manager() { AF_CHECK(af_create_memory_manager(&m_)); }

memory_manager::memory_manager(af_memory_manager m) : m_(m) {}

memory_manager::~memory_manager() {
    // No throw dtor
    af_release_memory_manager(m_);
}

memory_manager::memory_manager(memory_manager&& other) : m_(other.m_) {
    other.m_ = 0;
}

memory_manager& memory_manager::operator=(memory_manager&& other) {
    af_release_memory_manager(this->m_);
    this->m_ = other.m_;
    other.m_ = 0;
    return *this;
}

af_memory_manager memory_manager::get() const { return m_; }

void memory_manager::registerInitialize(InitializeFn fn) {
    AF_CHECK(af_memory_manager_set_initialize_fn(m_, fn));
}

void memory_manager::registerShutdown(ShutdownFn fn) {
    AF_CHECK(af_memory_manager_set_shutdown_fn(m_, fn));
}

void memory_manager::registerAlloc(AllocFn fn) {
    AF_CHECK(af_memory_manager_set_alloc_fn(m_, fn));
}

void memory_manager::registerAllocated(AllocatedFn fn) {
    AF_CHECK(af_memory_manager_set_allocated_fn(m_, fn));
}

void memory_manager::registerUnlock(UnlockFn fn) {
    AF_CHECK(af_memory_manager_set_unlock_fn(m_, fn));
}

void memory_manager::registerGarbageCollect(GarbageCollectFn fn) {
    AF_CHECK(af_memory_manager_set_garbage_collect_fn(m_, fn));
}

void memory_manager::registerPrintInfo(PrintInfoFn fn) {
    AF_CHECK(af_memory_manager_set_print_info_fn(m_, fn));
}

void memory_manager::registerUsageInfo(UsageInfoFn fn) {
    AF_CHECK(af_memory_manager_set_usage_info_fn(m_, fn));
}

void memory_manager::registerUserLock(UserLockFn fn) {
    AF_CHECK(af_memory_manager_set_user_lock_fn(m_, fn));
}

void memory_manager::registerUserUnlock(UserUnlockFn fn) {
    AF_CHECK(af_memory_manager_set_user_unlock_fn(m_, fn));
}

void memory_manager::registerIsUserLocked(IsUserLockedFn fn) {
    AF_CHECK(af_memory_manager_set_is_user_locked_fn(m_, fn));
}

void memory_manager::registerGetMemStepSize(GetMemStepSizeFn fn) {
    AF_CHECK(af_memory_manager_set_get_mem_step_size_fn(m_, fn));
}

void memory_manager::registerGetMaxBytes(GetMaxBytesFn fn) {
    AF_CHECK(af_memory_manager_set_get_max_bytes_fn(m_, fn));
}

void memory_manager::registerGetMaxBuffers(GetMaxBuffersFn fn) {
    AF_CHECK(af_memory_manager_set_get_max_buffers_fn(m_, fn));
}

void memory_manager::registerSetMemStepSize(SetMemStepSizeFn fn) {
    AF_CHECK(af_memory_manager_set_set_mem_step_size_fn(m_, fn));
}

void memory_manager::registerCheckMemoryLimit(CheckMemoryLimitFn fn) {
    AF_CHECK(af_memory_manager_set_check_memory_limit_fn(m_, fn));
}

void memory_manager::registerAddMemoryManagement(AddMemoryManagementFn fn) {
    AF_CHECK(af_memory_manager_set_add_memory_management_fn(m_, fn));
}

void memory_manager::registerRemoveMemoryManagement(
    RemoveMemoryManagementFn fn) {
    AF_CHECK(af_memory_manager_set_remove_memory_management_fn(m_, fn));
}

void memory_manager::setPayload(void* payload) {
    AF_CHECK(af_memory_manager_set_payload(m_, payload));
}

void* memory_manager::getPayload() const {
    void* payload;
    AF_CHECK(af_memory_manager_get_payload(m_, &payload));
    return payload;
}

}  // namespace af
