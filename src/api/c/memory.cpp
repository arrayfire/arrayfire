/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <memoryapi.hpp>
#include <platform.hpp>
#include <af/backend.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/memory.h>
#include <af/version.h>
#include <cstring>

using namespace detail;

using common::half;

af_memory_event_pair_t getMemoryEventPair(
    const af_memory_event_pair pairHandle) {
    return *(af_memory_event_pair_t *)pairHandle;
}

af_memory_event_pair getMemoryEventPairHandle(
    const af_memory_event_pair_t pair) {
    af_memory_event_pair_t *pairHandle = new af_memory_event_pair_t;
    *pairHandle                        = pair;
    return (af_memory_event_pair)pairHandle;
}

af_err af_create_memory_event_pair(af_memory_event_pair *pairHandle, void *ptr,
                                   af_event event) {
    try {
        af_memory_event_pair_t pair;
        pair.ptr    = ptr;
        pair.event  = event;
        *pairHandle = getMemoryEventPairHandle(pair);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_release_memory_event_pair(af_memory_event_pair pairHandle) {
    try {
        af_memory_event_pair_t pair = *(af_memory_event_pair_t *)pairHandle;
        /// NB: deleting a memory event pair does NOT free the associated memory
        /// and does NOT delete the associated event.
        delete (af_memory_event_pair_t *)pairHandle;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_event_pair_set_ptr(af_memory_event_pair pairHandle,
                                    void *ptr) {
    try {
        af_memory_event_pair_t *pair = (af_memory_event_pair_t *)pairHandle;
        pair->ptr                    = ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_event_pair_set_event(af_memory_event_pair pairHandle,
                                      af_event event) {
    try {
        af_memory_event_pair_t *pair = (af_memory_event_pair_t *)pairHandle;
        pair->event                  = event;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_event_pair_get_ptr(void **ptr,
                                    af_memory_event_pair pairHandle) {
    try {
        af_memory_event_pair_t pair = getMemoryEventPair(pairHandle);
        *ptr                        = pair.ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_memory_event_pair_get_event(af_event *event,
                                      af_memory_event_pair pairHandle) {
    try {
        af_memory_event_pair_t pair = getMemoryEventPair(pairHandle);
        *event                      = pair.event;
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
