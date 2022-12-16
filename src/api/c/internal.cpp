/*******************************************************
 * Copyright (c) 2016, ArrayFire
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
#include <platform.hpp>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/internal.h>
#include <af/version.h>
#include <cstring>

using af::dim4;
using arrayfire::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::createStridedArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

af_err af_create_strided_array(af_array *arr, const void *data,
                               const dim_t offset, const unsigned ndims,
                               const dim_t *const dims_,
                               const dim_t *const strides_, const af_dtype ty,
                               const af_source location) {
    try {
        ARG_ASSERT(2, offset >= 0);
        ARG_ASSERT(3, ndims >= 1 && ndims <= 4);
        ARG_ASSERT(4, dims_ != NULL);
        ARG_ASSERT(5, strides_ != NULL);
        ARG_ASSERT(5, strides_[0] == 1);

        for (int i = 1; i < static_cast<int>(ndims); i++) {
            ARG_ASSERT(5, strides_[i] > 0);
        }

        dim4 dims(ndims, dims_);
        dim4 strides(ndims, strides_);

        for (int i = static_cast<int>(ndims); i < 4; i++) {
            strides[i] = strides[i - 1] * dims[i - 1];
        }

        bool isdev = location == afDevice;

        af_array res;
        AF_CHECK(af_init());

        void *in_data = const_cast<void *>(
            data);  // const cast because the api cannot change
        switch (ty) {
            case f32:
                res = getHandle(createStridedArray<float>(
                    dims, strides, offset, static_cast<float *>(in_data),
                    isdev));
                break;
            case f64:
                res = getHandle(createStridedArray<double>(
                    dims, strides, offset, static_cast<double *>(in_data),
                    isdev));
                break;
            case c32:
                res = getHandle(createStridedArray<cfloat>(
                    dims, strides, offset, static_cast<cfloat *>(in_data),
                    isdev));
                break;
            case c64:
                res = getHandle(createStridedArray<cdouble>(
                    dims, strides, offset, static_cast<cdouble *>(in_data),
                    isdev));
                break;
            case u32:
                res = getHandle(createStridedArray<uint>(
                    dims, strides, offset, static_cast<uint *>(in_data),
                    isdev));
                break;
            case s32:
                res = getHandle(createStridedArray<int>(
                    dims, strides, offset, static_cast<int *>(in_data), isdev));
                break;
            case u64:
                res = getHandle(createStridedArray<uintl>(
                    dims, strides, offset, static_cast<uintl *>(in_data),
                    isdev));
                break;
            case s64:
                res = getHandle(createStridedArray<intl>(
                    dims, strides, offset, static_cast<intl *>(in_data),
                    isdev));
                break;
            case u16:
                res = getHandle(createStridedArray<ushort>(
                    dims, strides, offset, static_cast<ushort *>(in_data),
                    isdev));
                break;
            case s16:
                res = getHandle(createStridedArray<short>(
                    dims, strides, offset, static_cast<short *>(in_data),
                    isdev));
                break;
            case b8:
                res = getHandle(createStridedArray<char>(
                    dims, strides, offset, static_cast<char *>(in_data),
                    isdev));
                break;
            case u8:
                res = getHandle(createStridedArray<uchar>(
                    dims, strides, offset, static_cast<uchar *>(in_data),
                    isdev));
                break;
            case f16:
                res = getHandle(createStridedArray<half>(
                    dims, strides, offset, static_cast<half *>(in_data),
                    isdev));
                break;
            default: TYPE_ERROR(6, ty);
        }

        std::swap(*arr, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3,
                      const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        *s0                   = info.strides()[0];
        *s1                   = info.strides()[1];
        *s2                   = info.strides()[2];
        *s3                   = info.strides()[3];
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_offset(dim_t *offset, const af_array arr) {
    try {
        dim_t res = getInfo(arr).getOffset();
        std::swap(*offset, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_raw_ptr(void **ptr, const af_array arr) {
    try {
        void *res = NULL;

        af_dtype ty = getInfo(arr).getType();

        switch (ty) {
            case f32: res = getRawPtr(getArray<float>(arr)); break;
            case f64: res = getRawPtr(getArray<double>(arr)); break;
            case c32: res = getRawPtr(getArray<cfloat>(arr)); break;
            case c64: res = getRawPtr(getArray<cdouble>(arr)); break;
            case u32: res = getRawPtr(getArray<uint>(arr)); break;
            case s32: res = getRawPtr(getArray<int>(arr)); break;
            case u64: res = getRawPtr(getArray<uintl>(arr)); break;
            case s64: res = getRawPtr(getArray<intl>(arr)); break;
            case u16: res = getRawPtr(getArray<ushort>(arr)); break;
            case s16: res = getRawPtr(getArray<short>(arr)); break;
            case b8: res = getRawPtr(getArray<char>(arr)); break;
            case u8: res = getRawPtr(getArray<uchar>(arr)); break;
            case f16: res = getRawPtr(getArray<half>(arr)); break;
            default: TYPE_ERROR(6, ty);
        }

        std::swap(*ptr, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_is_linear(bool *result, const af_array arr) {
    try {
        *result = getInfo(arr).isLinear();
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_is_owner(bool *result, const af_array arr) {
    try {
        bool res = false;

        af_dtype ty = getInfo(arr).getType();

        switch (ty) {
            case f32: res = getArray<float>(arr).isOwner(); break;
            case f64: res = getArray<double>(arr).isOwner(); break;
            case c32: res = getArray<cfloat>(arr).isOwner(); break;
            case c64: res = getArray<cdouble>(arr).isOwner(); break;
            case u32: res = getArray<uint>(arr).isOwner(); break;
            case s32: res = getArray<int>(arr).isOwner(); break;
            case u64: res = getArray<uintl>(arr).isOwner(); break;
            case s64: res = getArray<intl>(arr).isOwner(); break;
            case u16: res = getArray<ushort>(arr).isOwner(); break;
            case s16: res = getArray<short>(arr).isOwner(); break;
            case b8: res = getArray<char>(arr).isOwner(); break;
            case u8: res = getArray<uchar>(arr).isOwner(); break;
            case f16: res = getArray<half>(arr).isOwner(); break;
            default: TYPE_ERROR(6, ty);
        }

        std::swap(*result, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_allocated_bytes(size_t *bytes, const af_array arr) {
    try {
        af_dtype ty = getInfo(arr).getType();

        size_t res = 0;

        switch (ty) {
            case f32: res = getArray<float>(arr).getAllocatedBytes(); break;
            case f64: res = getArray<double>(arr).getAllocatedBytes(); break;
            case c32: res = getArray<cfloat>(arr).getAllocatedBytes(); break;
            case c64: res = getArray<cdouble>(arr).getAllocatedBytes(); break;
            case u32: res = getArray<uint>(arr).getAllocatedBytes(); break;
            case s32: res = getArray<int>(arr).getAllocatedBytes(); break;
            case u64: res = getArray<uintl>(arr).getAllocatedBytes(); break;
            case s64: res = getArray<intl>(arr).getAllocatedBytes(); break;
            case u16: res = getArray<ushort>(arr).getAllocatedBytes(); break;
            case s16: res = getArray<short>(arr).getAllocatedBytes(); break;
            case b8: res = getArray<char>(arr).getAllocatedBytes(); break;
            case u8: res = getArray<uchar>(arr).getAllocatedBytes(); break;
            case f16: res = getArray<half>(arr).getAllocatedBytes(); break;
            default: TYPE_ERROR(6, ty);
        }

        std::swap(*bytes, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}
