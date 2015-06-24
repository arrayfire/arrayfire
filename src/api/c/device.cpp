/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/device.h>
#include <af/version.h>
#include <backend.hpp>
#include <platform.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include "err_common.hpp"

using namespace detail;

af_err af_init()
{
    try {
        static bool first = true;
        if(first) {
            getInfo();
            first = false;
        }
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_info()
{
    printf("%s", getInfo().c_str());
    return AF_SUCCESS;
}

af_err af_get_version(int *major, int *minor, int *patch)
{
    *major = AF_VERSION_MAJOR;
    *minor = AF_VERSION_MINOR;
    *patch = AF_VERSION_PATCH;

    return AF_SUCCESS;
}

af_err af_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    try {
        devprop(d_name, d_platform, d_toolkit, d_compute);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_dbl_support(bool* available, const int device)
{
    try {
        *available = isDoubleSupported(device);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_count(int *nDevices)
{
    try {
        *nDevices = getDeviceCount();
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device(int *device)
{
    try {
        *device = getActiveDeviceId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_device(const int device)
{
    try {
        ARG_ASSERT(0, device >= 0);
        ARG_ASSERT(0, setDevice(device) >= 0);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_sync(const int device)
{
    try {
        int dev = device == -1 ? getActiveDeviceId() : device;
        detail::sync(dev);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_device_array(af_array *arr, const void *data,
                       const unsigned ndims,
                       const dim_t * const dims,
                       const af_dtype type)
{
    try {
        AF_CHECK(af_init());

        af_array res;
        af::dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }

        switch (type) {
        case f32: res = getHandle(createDeviceDataArray<float  >(d, data)); break;
        case f64: res = getHandle(createDeviceDataArray<double >(d, data)); break;
        case c32: res = getHandle(createDeviceDataArray<cfloat >(d, data)); break;
        case c64: res = getHandle(createDeviceDataArray<cdouble>(d, data)); break;
        case s32: res = getHandle(createDeviceDataArray<int    >(d, data)); break;
        case u32: res = getHandle(createDeviceDataArray<uint   >(d, data)); break;
        case s64: res = getHandle(createDeviceDataArray<intl   >(d, data)); break;
        case u64: res = getHandle(createDeviceDataArray<uintl  >(d, data)); break;
        case u8 : res = getHandle(createDeviceDataArray<uchar  >(d, data)); break;
        case b8 : res = getHandle(createDeviceDataArray<char   >(d, data)); break;
        default: TYPE_ERROR(4, type);
        }

        std::swap(*arr, res);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device_ptr(void **data, const af_array arr)
{
    try {

        // Make sure all kernels and memcopies are done before getting device pointer
        detail::sync(getActiveDeviceId());

        af_dtype type = getInfo(arr).getType();

        switch (type) {
            //FIXME: Perform copy if memory not continuous
        case f32: *data = getDevicePtr(getArray<float  >(arr)); break;
        case f64: *data = getDevicePtr(getArray<double >(arr)); break;
        case c32: *data = getDevicePtr(getArray<cfloat >(arr)); break;
        case c64: *data = getDevicePtr(getArray<cdouble>(arr)); break;
        case s32: *data = getDevicePtr(getArray<int    >(arr)); break;
        case u32: *data = getDevicePtr(getArray<uint   >(arr)); break;
        case s64: *data = getDevicePtr(getArray<intl   >(arr)); break;
        case u64: *data = getDevicePtr(getArray<uintl  >(arr)); break;
        case u8 : *data = getDevicePtr(getArray<uchar  >(arr)); break;
        case b8 : *data = getDevicePtr(getArray<char   >(arr)); break;

        default: TYPE_ERROR(4, type);
        }

    } CATCHALL;

    return AF_SUCCESS;
}

template <typename T>
inline void lockDevicePtr(const af_array arr)
{
    memPop<T>((const T *)getArray<T>(arr).get());
}

af_err af_lock_device_ptr(const af_array arr)
{
    try {

        // Make sure all kernels and memcopies are done before getting device pointer
        detail::sync(getActiveDeviceId());

        af_dtype type = getInfo(arr).getType();

        switch (type) {
        case f32: lockDevicePtr<float  >(arr); break;
        case f64: lockDevicePtr<double >(arr); break;
        case c32: lockDevicePtr<cfloat >(arr); break;
        case c64: lockDevicePtr<cdouble>(arr); break;
        case s32: lockDevicePtr<int    >(arr); break;
        case u32: lockDevicePtr<uint   >(arr); break;
        case s64: lockDevicePtr<intl   >(arr); break;
        case u64: lockDevicePtr<uintl  >(arr); break;
        case u8 : lockDevicePtr<uchar  >(arr); break;
        case b8 : lockDevicePtr<char   >(arr); break;
        default: TYPE_ERROR(4, type);
        }

    } CATCHALL;

    return AF_SUCCESS;
}

template <typename T>
inline void unlockDevicePtr(const af_array arr)
{
    memPush<T>((const T *)getArray<T>(arr).get());
}

af_err af_unlock_device_ptr(const af_array arr)
{
    try {

        // Make sure all kernels and memcopies are done before getting device pointer
        detail::sync(getActiveDeviceId());

        af_dtype type = getInfo(arr).getType();

        switch (type) {
        case f32: unlockDevicePtr<float  >(arr); break;
        case f64: unlockDevicePtr<double >(arr); break;
        case c32: unlockDevicePtr<cfloat >(arr); break;
        case c64: unlockDevicePtr<cdouble>(arr); break;
        case s32: unlockDevicePtr<int    >(arr); break;
        case u32: unlockDevicePtr<uint   >(arr); break;
        case s64: unlockDevicePtr<intl   >(arr); break;
        case u64: unlockDevicePtr<uintl  >(arr); break;
        case u8 : unlockDevicePtr<uchar  >(arr); break;
        case b8 : unlockDevicePtr<char   >(arr); break;
        default: TYPE_ERROR(4, type);
        }

    } CATCHALL;

    return AF_SUCCESS;
}


af_err af_alloc_device(void **ptr, const dim_t bytes)
{
    try {
        AF_CHECK(af_init());
        *ptr = (void *)memAlloc<char>(bytes);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_alloc_pinned(void **ptr, const dim_t bytes)
{
    try {
        AF_CHECK(af_init());
        *ptr = (void *)pinnedAlloc<char>(bytes);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_free_device(void *ptr)
{
    try {
        memFree<char>((char *)ptr);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_free_pinned(void *ptr)
{
    try {
        pinnedFree<char>((char *)ptr);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_device_gc()
{
    try {
        garbageCollect();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes,  size_t *lock_buffers)
{
    try {
        deviceMemoryInfo(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_mem_step_size(const size_t step_bytes)
{
    detail::setMemStepSize(step_bytes);
    return AF_SUCCESS;
}

af_err af_get_mem_step_size(size_t *step_bytes)
{
    *step_bytes =  detail::getMemStepSize();
    return AF_SUCCESS;
}
