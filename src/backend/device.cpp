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
#include <backend.hpp>
#include <platform.hpp>
#include <iostream>
#include <Array.hpp>
#include <handle.hpp>
#include "err_common.hpp"

using namespace detail;

af_err af_info()
{
    std::cout << getInfo();
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
        if (setDevice(device) < 0) {
            std::cout << "Invalid Device ID" << std::endl;
            return AF_ERR_INVALID_ARG;
        }
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
                       const dim_type * const dims,
                       const af_dtype type)
{
    try {

        af_array res;
        af::dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }

        switch (type) {
        case f32: res = getHandle(*createDeviceDataArray<float  >(d, data)); break;
        case f64: res = getHandle(*createDeviceDataArray<double >(d, data)); break;
        case c32: res = getHandle(*createDeviceDataArray<cfloat >(d, data)); break;
        case c64: res = getHandle(*createDeviceDataArray<cdouble>(d, data)); break;
        case s32: res = getHandle(*createDeviceDataArray<int    >(d, data)); break;
        case u32: res = getHandle(*createDeviceDataArray<uint   >(d, data)); break;
        case u8 : res = getHandle(*createDeviceDataArray<uchar  >(d, data)); break;
        case b8 : res = getHandle(*createDeviceDataArray<char   >(d, data)); break;
        default: TYPE_ERROR(4, type);
        }

        std::swap(*arr, res);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device_ptr(void *data, const af_array arr, bool read_only)
{
    if (!read_only) {
        //FIXME: Implement a lock / unlock mechanism
        AF_ERROR("Write access to device pointer not yet implemented", AF_ERR_NOT_SUPPORTED);
    }

    try {
        af_dtype type = getInfo(arr).getType();

        switch (type) {
            //FIXME: Perform copy if memory not continuous
        case f32: data = getDevicePtr(getArray<float  >(arr)); break;
        case f64: data = getDevicePtr(getArray<double >(arr)); break;
        case c32: data = getDevicePtr(getArray<cfloat >(arr)); break;
        case c64: data = getDevicePtr(getArray<cdouble>(arr)); break;
        case s32: data = getDevicePtr(getArray<int    >(arr)); break;
        case u32: data = getDevicePtr(getArray<uint   >(arr)); break;
        case u8 : data = getDevicePtr(getArray<uchar  >(arr)); break;
        case b8 : data = getDevicePtr(getArray<char   >(arr)); break;
        default: TYPE_ERROR(4, type);
        }
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_alloc_device(void **ptr, dim_type bytes)
{
    AF_ERROR("Memory manager not yet implemented", AF_ERR_NOT_SUPPORTED);
    return AF_SUCCESS;
}

af_err af_alloc_pinned(void **ptr, dim_type bytes)
{
    AF_ERROR("Memory manager not yet implemented", AF_ERR_NOT_SUPPORTED);
    return AF_SUCCESS;
}

af_err af_free_device(void *ptr)
{
    AF_ERROR("Memory manager not yet implemented", AF_ERR_NOT_SUPPORTED);
    return AF_SUCCESS;
}

af_err af_free_pinned(void *ptr)
{
    AF_ERROR("Memory manager not yet implemented", AF_ERR_NOT_SUPPORTED);
    return AF_SUCCESS;
}
