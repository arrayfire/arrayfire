/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <optypes.hpp>
#include <scan.hpp>
#include <scan_by_key.hpp>
#include <af/algorithm.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <complex>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<af_op_t op, typename Ti, typename To>
static inline af_array scan(const af_array in, const int dim,
                            bool inclusive_scan = true) {
    return getHandle(scan<op, Ti, To>(getArray<Ti>(in), dim, inclusive_scan));
}

template<af_op_t op, typename Ti, typename To>
static inline af_array scan_key(const af_array key, const af_array in,
                                const int dim, bool inclusive_scan = true) {
    const ArrayInfo& key_info = getInfo(key);
    af_dtype type             = key_info.getType();
    af_array out;

    switch (type) {
        case s32:
            out = getHandle(scan<op, Ti, int, To>(
                getArray<int>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case u32:
            out = getHandle(scan<op, Ti, uint, To>(
                getArray<uint>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case s64:
            out = getHandle(scan<op, Ti, intl, To>(
                getArray<intl>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case u64:
            out = getHandle(scan<op, Ti, uintl, To>(
                getArray<uintl>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        default: TYPE_ERROR(1, type);
    }
    return out;
}

template<typename Ti, typename To>
static inline af_array scan_op(const af_array key, const af_array in,
                               const int dim, af_binary_op op,
                               bool inclusive_scan = true) {
    af_array out;

    switch (op) {
        case AF_BINARY_ADD:
            out = scan_key<af_add_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case AF_BINARY_MUL:
            out = scan_key<af_mul_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case AF_BINARY_MIN:
            out = scan_key<af_min_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case AF_BINARY_MAX:
            out = scan_key<af_max_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        default:
            AF_ERROR("Incorrect binary operation enum for argument number 3",
                     AF_ERR_ARG);
            break;
    }
    return out;
}

template<typename Ti, typename To>
static inline af_array scan_op(const af_array in, const int dim,
                               af_binary_op op, bool inclusive_scan) {
    af_array out;

    switch (op) {
        case AF_BINARY_ADD:
            out = scan<af_add_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case AF_BINARY_MUL:
            out = scan<af_mul_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case AF_BINARY_MIN:
            out = scan<af_min_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case AF_BINARY_MAX:
            out = scan<af_max_t, Ti, To>(in, dim, inclusive_scan);
            break;
        default:
            AF_ERROR("Incorrect binary operation enum for argument number 2",
                     AF_ERR_ARG);
            break;
    }
    return out;
}

af_err af_accum(af_array* out, const af_array in, const int dim) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32: res = scan<af_add_t, float, float>(in, dim); break;
            case f64: res = scan<af_add_t, double, double>(in, dim); break;
            case c32: res = scan<af_add_t, cfloat, cfloat>(in, dim); break;
            case c64: res = scan<af_add_t, cdouble, cdouble>(in, dim); break;
            case u32: res = scan<af_add_t, uint, uint>(in, dim); break;
            case s32: res = scan<af_add_t, int, int>(in, dim); break;
            case u64: res = scan<af_add_t, uintl, uintl>(in, dim); break;
            case s64: res = scan<af_add_t, intl, intl>(in, dim); break;
            case u16: res = scan<af_add_t, ushort, uint>(in, dim); break;
            case s16: res = scan<af_add_t, short, int>(in, dim); break;
            case u8: res = scan<af_add_t, uchar, uint>(in, dim); break;
            // Make sure you are adding only "1" for every non zero value, even
            // if op == af_add_t
            case b8: res = scan<af_notzero_t, char, uint>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_scan(af_array* out, const af_array in, const int dim, af_binary_op op,
               bool inclusive_scan) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32:
                res = scan_op<float, float>(in, dim, op, inclusive_scan);
                break;
            case f64:
                res = scan_op<double, double>(in, dim, op, inclusive_scan);
                break;
            case c32:
                res = scan_op<cfloat, cfloat>(in, dim, op, inclusive_scan);
                break;
            case c64:
                res = scan_op<cdouble, cdouble>(in, dim, op, inclusive_scan);
                break;
            case u32:
                res = scan_op<uint, uint>(in, dim, op, inclusive_scan);
                break;
            case s32:
                res = scan_op<int, int>(in, dim, op, inclusive_scan);
                break;
            case u64:
                res = scan_op<uintl, uintl>(in, dim, op, inclusive_scan);
                break;
            case s64:
                res = scan_op<intl, intl>(in, dim, op, inclusive_scan);
                break;
            case u16:
                res = scan_op<ushort, uint>(in, dim, op, inclusive_scan);
                break;
            case s16:
                res = scan_op<short, int>(in, dim, op, inclusive_scan);
                break;
            case u8:
                res = scan_op<uchar, uint>(in, dim, op, inclusive_scan);
                break;
            case b8:
                res = scan_op<char, uint>(in, dim, op, inclusive_scan);
                break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_scan_by_key(af_array* out, const af_array key, const af_array in,
                      const int dim, af_binary_op op, bool inclusive_scan) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info  = getInfo(in);
        const ArrayInfo& key_info = getInfo(key);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        ARG_ASSERT(2, in_info.dims() == key_info.dims());

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32:
                res = scan_op<float, float>(key, in, dim, op, inclusive_scan);
                break;
            case f64:
                res = scan_op<double, double>(key, in, dim, op, inclusive_scan);
                break;
            case c32:
                res = scan_op<cfloat, cfloat>(key, in, dim, op, inclusive_scan);
                break;
            case c64:
                res =
                    scan_op<cdouble, cdouble>(key, in, dim, op, inclusive_scan);
                break;
            case s16:
            case s32:
                res = scan_op<int, int>(key, in, dim, op, inclusive_scan);
                break;
            case u64:
                res = scan_op<uintl, uintl>(key, in, dim, op, inclusive_scan);
                break;
            case s64:
                res = scan_op<intl, intl>(key, in, dim, op, inclusive_scan);
                break;
            case u16:
            case u32:
            case u8:
            case b8:
                res = scan_op<uint, uint>(key, in, dim, op, inclusive_scan);
                break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
