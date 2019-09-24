/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memoryapi.hpp>

#include <Array.hpp>
#include <backend.hpp>
#include <common/MemoryManager.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <af/backend.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/memory.h>
#include <af/version.h>

#include <utility>

using namespace detail;

using common::half;

BufferInfo &getBufferInfo(const af_buffer_info handle) {
    return *(BufferInfo *)handle;
}

af_buffer_info getHandle(BufferInfo &buf) {
    BufferInfo *handle;
    handle = &buf;
    return (af_buffer_info)handle;
}

detail::Event &getEventFromBufferInfoHandle(const af_buffer_info handle) {
    return getEvent(getBufferInfo(handle).event);
}

af_err af_create_buffer_info(af_buffer_info *handle, void *ptr,
                             af_event event) {
    try {
        BufferInfo *buf = new BufferInfo({ptr, event});
        *handle         = getHandle(*buf);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_delete_buffer_info(af_buffer_info handle) {
    try {
        /// NB: deleting a memory event buf does frees the associated memory
        /// and deletes the associated event. Use unlock functions to free
        /// resources individually
        BufferInfo &buf = getBufferInfo(handle);
        af_release_event(buf.event);
        if (buf.ptr) { af_free_device(buf.ptr); }

        delete (BufferInfo *)handle;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_get_ptr(void **ptr, af_buffer_info handle) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        *ptr            = buf.ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_get_event(af_event *event, af_buffer_info handle) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        *event          = buf.event;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_set_ptr(af_buffer_info handle, void *ptr) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        buf.ptr         = ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_set_event(af_buffer_info handle, af_event event) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        buf.event       = event;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_unlock_buffer_info_event(af_event *event, af_buffer_info handle) {
    try {
        af_buffer_info_get_event(event, handle);
        BufferInfo &buf = getBufferInfo(handle);
        buf.event       = 0;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_unlock_buffer_info_ptr(void **ptr, af_buffer_info handle) {
    try {
        af_buffer_info_get_ptr(ptr, handle);
        BufferInfo &buf = getBufferInfo(handle);
        buf.ptr         = 0;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_device_array(af_array *arr, void *data, const unsigned ndims,
                       const dim_t *const dims, const af_dtype type) {
    try {
        AF_CHECK(af_init());

        af_array res;

        DIM_ASSERT(1, ndims >= 1);
        dim4 d(1, 1, 1, 1);
        for (unsigned i = 0; i < ndims; i++) {
            d[i] = dims[i];
            DIM_ASSERT(3, dims[i] >= 1);
        }

        switch (type) {
            case f32:
                res = getHandle(createDeviceDataArray<float>(d, data));
                break;
            case f64:
                res = getHandle(createDeviceDataArray<double>(d, data));
                break;
            case c32:
                res = getHandle(createDeviceDataArray<cfloat>(d, data));
                break;
            case c64:
                res = getHandle(createDeviceDataArray<cdouble>(d, data));
                break;
            case s32:
                res = getHandle(createDeviceDataArray<int>(d, data));
                break;
            case u32:
                res = getHandle(createDeviceDataArray<uint>(d, data));
                break;
            case s64:
                res = getHandle(createDeviceDataArray<intl>(d, data));
                break;
            case u64:
                res = getHandle(createDeviceDataArray<uintl>(d, data));
                break;
            case s16:
                res = getHandle(createDeviceDataArray<short>(d, data));
                break;
            case u16:
                res = getHandle(createDeviceDataArray<ushort>(d, data));
                break;
            case u8:
                res = getHandle(createDeviceDataArray<uchar>(d, data));
                break;
            case b8:
                res = getHandle(createDeviceDataArray<char>(d, data));
                break;
            case f16:
                res = getHandle(createDeviceDataArray<half>(d, data));
                break;
            default: TYPE_ERROR(4, type);
        }

        std::swap(*arr, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device_ptr(void **data, const af_array arr) {
    try {
        af_dtype type = getInfo(arr).getType();

        switch (type) {
            // FIXME: Perform copy if memory not continuous
            case f32: *data = getDevicePtr(getArray<float>(arr)); break;
            case f64: *data = getDevicePtr(getArray<double>(arr)); break;
            case c32: *data = getDevicePtr(getArray<cfloat>(arr)); break;
            case c64: *data = getDevicePtr(getArray<cdouble>(arr)); break;
            case s32: *data = getDevicePtr(getArray<int>(arr)); break;
            case u32: *data = getDevicePtr(getArray<uint>(arr)); break;
            case s64: *data = getDevicePtr(getArray<intl>(arr)); break;
            case u64: *data = getDevicePtr(getArray<uintl>(arr)); break;
            case s16: *data = getDevicePtr(getArray<short>(arr)); break;
            case u16: *data = getDevicePtr(getArray<ushort>(arr)); break;
            case u8: *data = getDevicePtr(getArray<uchar>(arr)); break;
            case b8: *data = getDevicePtr(getArray<char>(arr)); break;
            case f16: *data = getDevicePtr(getArray<half>(arr)); break;

            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
inline void lockArray(const af_array arr) {
    // Ideally we need to use .get(false), i.e. get ptr without offset
    // This is however not supported in opencl
    // Use getData().get() as alternative
    memLock((void *)getArray<T>(arr).getData().get());
}

af_err af_lock_device_ptr(const af_array arr) { return af_lock_array(arr); }

af_err af_lock_array(const af_array arr) {
    try {
        af_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: lockArray<float>(arr); break;
            case f64: lockArray<double>(arr); break;
            case c32: lockArray<cfloat>(arr); break;
            case c64: lockArray<cdouble>(arr); break;
            case s32: lockArray<int>(arr); break;
            case u32: lockArray<uint>(arr); break;
            case s64: lockArray<intl>(arr); break;
            case u64: lockArray<uintl>(arr); break;
            case s16: lockArray<short>(arr); break;
            case u16: lockArray<ushort>(arr); break;
            case u8: lockArray<uchar>(arr); break;
            case b8: lockArray<char>(arr); break;
            case f16: lockArray<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
inline bool checkUserLock(const af_array arr) {
    // Ideally we need to use .get(false), i.e. get ptr without offset
    // This is however not supported in opencl
    // Use getData().get() as alternative
    return isLocked((void *)getArray<T>(arr).getData().get());
}

af_err af_is_locked_array(bool *res, const af_array arr) {
    try {
        af_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: *res = checkUserLock<float>(arr); break;
            case f64: *res = checkUserLock<double>(arr); break;
            case c32: *res = checkUserLock<cfloat>(arr); break;
            case c64: *res = checkUserLock<cdouble>(arr); break;
            case s32: *res = checkUserLock<int>(arr); break;
            case u32: *res = checkUserLock<uint>(arr); break;
            case s64: *res = checkUserLock<intl>(arr); break;
            case u64: *res = checkUserLock<uintl>(arr); break;
            case s16: *res = checkUserLock<short>(arr); break;
            case u16: *res = checkUserLock<ushort>(arr); break;
            case u8: *res = checkUserLock<uchar>(arr); break;
            case b8: *res = checkUserLock<char>(arr); break;
            case f16: *res = checkUserLock<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
inline void unlockArray(const af_array arr) {
    // Ideally we need to use .get(false), i.e. get ptr without offset
    // This is however not supported in opencl
    // Use getData().get() as alternative
    memUnlock((void *)getArray<T>(arr).getData().get());
}

af_err af_unlock_device_ptr(const af_array arr) { return af_unlock_array(arr); }

af_err af_unlock_array(const af_array arr) {
    try {
        af_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: unlockArray<float>(arr); break;
            case f64: unlockArray<double>(arr); break;
            case c32: unlockArray<cfloat>(arr); break;
            case c64: unlockArray<cdouble>(arr); break;
            case s32: unlockArray<int>(arr); break;
            case u32: unlockArray<uint>(arr); break;
            case s64: unlockArray<intl>(arr); break;
            case u64: unlockArray<uintl>(arr); break;
            case s16: unlockArray<short>(arr); break;
            case u16: unlockArray<ushort>(arr); break;
            case u8: unlockArray<uchar>(arr); break;
            case b8: unlockArray<char>(arr); break;
            case f16: unlockArray<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_alloc_device(void **ptr, const dim_t bytes) {
    try {
        AF_CHECK(af_init());
        *ptr = memAllocUser(bytes);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_alloc_pinned(void **ptr, const dim_t bytes) {
    try {
        AF_CHECK(af_init());
        *ptr = (void *)pinnedAlloc<char>(bytes);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_free_device(void *ptr) {
    try {
        memFreeUser(ptr);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_free_pinned(void *ptr) {
    try {
        pinnedFree<char>((char *)ptr);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_alloc_host(void **ptr, const dim_t bytes) {
    if ((*ptr = malloc(bytes))) { return AF_SUCCESS; }
    return AF_ERR_NO_MEM;
}

af_err af_free_host(void *ptr) {
    free(ptr);
    return AF_SUCCESS;
}

af_err af_print_mem_info(const char *msg, const int device_id) {
    try {
        int device = device_id;
        if (device == -1) { device = getActiveDeviceId(); }

        if (msg != NULL)
            ARG_ASSERT(0, strlen(msg) < 256);  // 256 character limit on msg
        ARG_ASSERT(1, device >= 0 && device < getDeviceCount());

        printMemInfo(msg ? msg : "", device);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_device_gc() {
    try {
        garbageCollect();
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes, size_t *lock_buffers) {
    try {
        deviceMemoryInfo(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_mem_step_size(const size_t step_bytes) {
    try {
        detail::setMemStepSize(step_bytes);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_mem_step_size(size_t *step_bytes) {
    try {
        *step_bytes = detail::getMemStepSize();
    }
    CATCHALL;
    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

MemoryManager &getMemoryManager(const af_memory_manager handle) {
    return *(MemoryManager *)handle;
}

af_memory_manager getHandle(MemoryManager &manager) {
    MemoryManager *handle;
    handle = &manager;
    return (af_memory_manager)handle;
}

af_err af_create_memory_manager(af_memory_manager *manager) {
    try {
        AF_CHECK(af_init());
        std::unique_ptr<MemoryManager> m;
        m.reset(new MemoryManager());
        MemoryManager &ref = *m.release();
        *manager           = getHandle(ref);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_release_memory_manager(af_memory_manager handle) {
    try {
        detail::resetMemoryManager();
        delete (MemoryManager *)handle;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_set_memory_manager(af_memory_manager mgr) {
    try {
        // NB: does NOT free if a non-default implementation is set as the
        // current memory manager - the user is responsible for freeing any
        // controlled memory
        std::unique_ptr<MemoryManagerFunctionWrapper> newManager;
        newManager.reset(new MemoryManagerFunctionWrapper(mgr));

        // Calls shutdown() on the existing memory manager
        detail::setMemoryManager(std::move(newManager));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_release_memory_manager_pinned(af_memory_manager handle) {
    try {
        detail::resetMemoryManagerPinned();
        delete (MemoryManager *)handle;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_set_memory_manager_pinned(af_memory_manager mgr) {
    try {
        // NB: does NOT free if a non-default implementation is set as the
        // current memory manager - the user is responsible for freeing any
        // controlled memory
        std::unique_ptr<MemoryManagerFunctionWrapper> newManager;
        newManager.reset(new MemoryManagerFunctionWrapper(mgr));

        // Calls shutdown() on the existing memory manager
        detail::setMemoryManagerPinned(std::move(newManager));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_get_payload(af_memory_manager handle, void **payload) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *payload               = manager.payload;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_payload(af_memory_manager handle, void *payload) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.payload        = payload;
    }
    CATCHALL;

    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Native memory interface wrapper implementations

af_err af_memory_manager_get_active_device_id(af_memory_manager handle,
                                              int *id) {
    try {
        *id = memoryManager().getActiveDeviceId();
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_native_alloc(af_memory_manager handle, void **ptr,
                                      size_t size) {
    try {
        *ptr = memoryManager().nativeAlloc(size);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_native_free(af_memory_manager handle, void *ptr) {
    try {
        memoryManager().nativeFree(ptr);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_get_max_memory_size(af_memory_manager handle,
                                             size_t *size, int id) {
    try {
        *size = memoryManager().getMaxMemorySize(id);
    }
    CATCHALL;

    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Function setters

af_err af_memory_manager_set_initialize_fn(af_memory_manager handle,
                                           af_memory_manager_initialize_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.initialize_fn  = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_shutdown_fn(af_memory_manager handle,
                                         af_memory_manager_shutdown_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.shutdown_fn    = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_alloc_fn(af_memory_manager handle,
                                      af_memory_manager_alloc_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.alloc_fn       = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_allocated_fn(af_memory_manager handle,
                                          af_memory_manager_allocated_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.allocated_fn   = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_unlock_fn(af_memory_manager handle,
                                       af_memory_manager_unlock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.unlock_fn      = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_garbage_collect_fn(
    af_memory_manager handle, af_memory_manager_garbage_collect_fn fn) {
    try {
        MemoryManager &manager     = getMemoryManager(handle);
        manager.garbage_collect_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_print_info_fn(af_memory_manager handle,
                                           af_memory_manager_print_info_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.print_info_fn  = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_usage_info_fn(af_memory_manager handle,
                                           af_memory_manager_usage_info_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.usage_info_fn  = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_user_lock_fn(af_memory_manager handle,
                                          af_memory_manager_user_lock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.user_lock_fn   = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_user_unlock_fn(
    af_memory_manager handle, af_memory_manager_user_unlock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.user_unlock_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_is_user_locked_fn(
    af_memory_manager handle, af_memory_manager_is_user_locked_fn fn) {
    try {
        MemoryManager &manager    = getMemoryManager(handle);
        manager.is_user_locked_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_get_mem_step_size_fn(
    af_memory_manager handle, af_memory_manager_get_mem_step_size_fn fn) {
    try {
        MemoryManager &manager       = getMemoryManager(handle);
        manager.get_mem_step_size_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_get_max_bytes_fn(
    af_memory_manager handle, af_memory_manager_get_max_bytes_fn fn) {
    try {
        MemoryManager &manager   = getMemoryManager(handle);
        manager.get_max_bytes_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_get_max_buffers_fn(
    af_memory_manager handle, af_memory_manager_get_max_buffers_fn fn) {
    try {
        MemoryManager &manager     = getMemoryManager(handle);
        manager.get_max_buffers_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_set_mem_step_size_fn(
    af_memory_manager handle, af_memory_manager_set_mem_step_size_fn fn) {
    try {
        MemoryManager &manager       = getMemoryManager(handle);
        manager.set_mem_step_size_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_check_memory_limit_fn(
    af_memory_manager handle, af_memory_manager_check_memory_limit fn) {
    try {
        MemoryManager &manager        = getMemoryManager(handle);
        manager.check_memory_limit_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_add_memory_management_fn(
    af_memory_manager handle, af_memory_manager_add_memory_management fn) {
    try {
        MemoryManager &manager           = getMemoryManager(handle);
        manager.add_memory_management_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_manager_set_remove_memory_management_fn(
    af_memory_manager handle, af_memory_manager_remove_memory_management fn) {
    try {
        MemoryManager &manager              = getMemoryManager(handle);
        manager.remove_memory_management_fn = fn;
    }
    CATCHALL;

    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory Manager wrapper implementations

void MemoryManagerFunctionWrapper::initialize() {
    getMemoryManager(handle_).initialize_fn(handle_);
}

void MemoryManagerFunctionWrapper::shutdown() {
    getMemoryManager(handle_).shutdown_fn(handle_);
}

af_buffer_info MemoryManagerFunctionWrapper::alloc(const size_t size,
                                                   bool user_lock) {
    return getMemoryManager(handle_).alloc_fn(handle_, size, (int)user_lock);
}

size_t MemoryManagerFunctionWrapper::allocated(void *ptr) {
    return getMemoryManager(handle_).allocated_fn(handle_, ptr);
}

void MemoryManagerFunctionWrapper::unlock(void *ptr, af_event e,
                                          bool user_unlock) {
    getMemoryManager(handle_).unlock_fn(handle_, ptr, e, (int)user_unlock);
}

void MemoryManagerFunctionWrapper::garbageCollect() {
    getMemoryManager(handle_).garbage_collect_fn(handle_);
}

void MemoryManagerFunctionWrapper::printInfo(const char *msg,
                                             const int device) {
    getMemoryManager(handle_).print_info_fn(handle_, const_cast<char *>(msg),
                                            device);
}

void MemoryManagerFunctionWrapper::usageInfo(size_t *alloc_bytes,
                                             size_t *alloc_buffers,
                                             size_t *lock_bytes,
                                             size_t *lock_buffers) {
    getMemoryManager(handle_).usage_info_fn(handle_, alloc_bytes, alloc_buffers,
                                            lock_bytes, lock_buffers);
}

void MemoryManagerFunctionWrapper::userLock(const void *ptr) {
    getMemoryManager(handle_).user_lock_fn(handle_, const_cast<void *>(ptr));
}

void MemoryManagerFunctionWrapper::userUnlock(const void *ptr) {
    getMemoryManager(handle_).user_unlock_fn(handle_, const_cast<void *>(ptr));
}

bool MemoryManagerFunctionWrapper::isUserLocked(const void *ptr) {
    return getMemoryManager(handle_).is_user_locked_fn(handle_,
                                                       const_cast<void *>(ptr));
}

size_t MemoryManagerFunctionWrapper::getMemStepSize() {
    return getMemoryManager(handle_).get_mem_step_size_fn(handle_);
}

size_t MemoryManagerFunctionWrapper::getMaxBytes() {
    return getMemoryManager(handle_).get_max_bytes_fn(handle_);
}

unsigned MemoryManagerFunctionWrapper::getMaxBuffers() {
    return getMemoryManager(handle_).get_max_buffers_fn(handle_);
}

void MemoryManagerFunctionWrapper::setMemStepSize(size_t new_step_size) {
    getMemoryManager(handle_).set_mem_step_size_fn(handle_, new_step_size);
}

bool MemoryManagerFunctionWrapper::checkMemoryLimit() {
    return getMemoryManager(handle_).check_memory_limit_fn(handle_);
}

void MemoryManagerFunctionWrapper::addMemoryManagement(int device) {
    getMemoryManager(handle_).add_memory_management_fn(handle_, device);
}

void MemoryManagerFunctionWrapper::removeMemoryManagement(int device) {
    getMemoryManager(handle_).remove_memory_management_fn(handle_, device);
}
