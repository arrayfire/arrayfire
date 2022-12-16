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
#include <common/half.hpp>
#include <handle.hpp>
#include <ireduce.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <reduce.hpp>
#include <af/algorithm.h>
#include <af/defines.h>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::imag;
using detail::intl;
using detail::real;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<af_op_t op, typename Ti, typename To>
static inline af_array reduce(const af_array in, const int dim,
                              bool change_nan = false, double nanval = 0) {
    return getHandle(
        reduce<op, Ti, To>(getArray<Ti>(in), dim, change_nan, nanval));
}

template<af_op_t op, typename Ti, typename Tk, typename To>
static inline void reduce_by_key(af_array *keys_out, af_array *vals_out,
                                 const af_array keys, const af_array vals,
                                 const int dim, bool change_nan,
                                 double nanval) {
    Array<Tk> oKeyArray = createEmptyArray<Tk>(dim4());
    Array<To> oValArray = createEmptyArray<To>(dim4());

    reduce_by_key<op, Ti, Tk, To>(oKeyArray, oValArray, getArray<Tk>(keys),
                                  getArray<Ti>(vals), dim, change_nan, nanval);

    *keys_out = getHandle(oKeyArray);
    *vals_out = getHandle(oValArray);
}

template<af_op_t op, typename Ti, typename To>
static inline void reduce_key(af_array *keys_out, af_array *vals_out,
                              const af_array keys, const af_array vals,
                              const int dim, bool change_nan = false,
                              double nanval = 0.0) {
    const ArrayInfo &key_info = getInfo(keys);
    af_dtype type             = key_info.getType();

    switch (type) {
        case s32:
            reduce_by_key<op, Ti, int, To>(keys_out, vals_out, keys, vals, dim,
                                           change_nan, nanval);
            break;
        case u32:
            reduce_by_key<op, Ti, uint, To>(keys_out, vals_out, keys, vals, dim,
                                            change_nan, nanval);
            break;
        default: TYPE_ERROR(2, type);
    }
}

template<af_op_t op, typename To>
static af_err reduce_type(af_array *out, const af_array in, const int dim) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo &in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32: res = reduce<op, float, To>(in, dim); break;
            case f64: res = reduce<op, double, To>(in, dim); break;
            case c32: res = reduce<op, cfloat, To>(in, dim); break;
            case c64: res = reduce<op, cdouble, To>(in, dim); break;
            case u32: res = reduce<op, uint, To>(in, dim); break;
            case s32: res = reduce<op, int, To>(in, dim); break;
            case u64: res = reduce<op, uintl, To>(in, dim); break;
            case s64: res = reduce<op, intl, To>(in, dim); break;
            case u16: res = reduce<op, ushort, To>(in, dim); break;
            case s16: res = reduce<op, short, To>(in, dim); break;
            case b8: res = reduce<op, char, To>(in, dim); break;
            case u8: res = reduce<op, uchar, To>(in, dim); break;
            case f16: res = reduce<op, half, To>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op, typename To>
static af_err reduce_by_key_type(af_array *keys_out, af_array *vals_out,
                                 const af_array keys, const af_array vals,
                                 const int dim) {
    try {
        ARG_ASSERT(4, dim >= 0);
        ARG_ASSERT(4, dim < 4);

        const ArrayInfo &kinfo   = getInfo(keys);
        const ArrayInfo &in_info = getInfo(vals);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(2, kinfo.isVector());
        ARG_ASSERT(2, in_info.dims()[dim] == kinfo.elements());

        switch (type) {
            case f32:
                reduce_key<op, float, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case f64:
                reduce_key<op, double, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case c32:
                reduce_key<op, cfloat, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case c64:
                reduce_key<op, cdouble, To>(keys_out, vals_out, keys, vals,
                                            dim);
                break;
            case u32:
                reduce_key<op, uint, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case s32:
                reduce_key<op, int, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case u64:
                reduce_key<op, uintl, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case s64:
                reduce_key<op, intl, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case u16:
                reduce_key<op, ushort, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case s16:
                reduce_key<op, short, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case b8:
                reduce_key<op, char, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case u8:
                reduce_key<op, uchar, To>(keys_out, vals_out, keys, vals, dim);
                break;
            case f16:
                reduce_key<op, half, To>(keys_out, vals_out, keys, vals, dim);
                break;
            default: TYPE_ERROR(3, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_common(af_array *out, const af_array in, const int dim) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo &in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            return af_retain_array(out, in);
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32: res = reduce<op, float, float>(in, dim); break;
            case f64: res = reduce<op, double, double>(in, dim); break;
            case c32: res = reduce<op, cfloat, cfloat>(in, dim); break;
            case c64: res = reduce<op, cdouble, cdouble>(in, dim); break;
            case u32: res = reduce<op, uint, uint>(in, dim); break;
            case s32: res = reduce<op, int, int>(in, dim); break;
            case u64: res = reduce<op, uintl, uintl>(in, dim); break;
            case s64: res = reduce<op, intl, intl>(in, dim); break;
            case u16: res = reduce<op, ushort, ushort>(in, dim); break;
            case s16: res = reduce<op, short, short>(in, dim); break;
            case b8: res = reduce<op, char, char>(in, dim); break;
            case u8: res = reduce<op, uchar, uchar>(in, dim); break;
            case f16: res = reduce<op, half, half>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_by_key_common(af_array *keys_out, af_array *vals_out,
                                   const af_array keys, const af_array vals,
                                   const int dim) {
    try {
        ARG_ASSERT(4, dim >= 0);
        ARG_ASSERT(4, dim < 4);

        const ArrayInfo &kinfo   = getInfo(keys);
        const ArrayInfo &in_info = getInfo(vals);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(2, kinfo.isVector());
        ARG_ASSERT(2, in_info.dims()[dim] == kinfo.dims()[0]);

        switch (type) {
            case f32:
                reduce_key<op, float, float>(keys_out, vals_out, keys, vals,
                                             dim);
                break;
            case f64:
                reduce_key<op, double, double>(keys_out, vals_out, keys, vals,
                                               dim);
                break;
            case c32:
                reduce_key<op, cfloat, cfloat>(keys_out, vals_out, keys, vals,
                                               dim);
                break;
            case c64:
                reduce_key<op, cdouble, cdouble>(keys_out, vals_out, keys, vals,
                                                 dim);
                break;
            case u32:
                reduce_key<op, uint, uint>(keys_out, vals_out, keys, vals, dim);
                break;
            case s32:
                reduce_key<op, int, int>(keys_out, vals_out, keys, vals, dim);
                break;
            case u64:
                reduce_key<op, uintl, uintl>(keys_out, vals_out, keys, vals,
                                             dim);
                break;
            case s64:
                reduce_key<op, intl, intl>(keys_out, vals_out, keys, vals, dim);
                break;
            case u16:
                reduce_key<op, ushort, ushort>(keys_out, vals_out, keys, vals,
                                               dim);
                break;
            case s16:
                reduce_key<op, short, short>(keys_out, vals_out, keys, vals,
                                             dim);
                break;
            case b8:
                reduce_key<op, char, char>(keys_out, vals_out, keys, vals, dim);
                break;
            case u8:
                reduce_key<op, uchar, uchar>(keys_out, vals_out, keys, vals,
                                             dim);
            case f16:
                reduce_key<op, half, half>(keys_out, vals_out, keys, vals, dim);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_promote(af_array *out, const af_array in, const int dim,
                             bool change_nan = false, double nanval = 0.0) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo &in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch (type) {
            case f32:
                res = reduce<op, float, float>(in, dim, change_nan, nanval);
                break;
            case f64:
                res = reduce<op, double, double>(in, dim, change_nan, nanval);
                break;
            case c32:
                res = reduce<op, cfloat, cfloat>(in, dim, change_nan, nanval);
                break;
            case c64:
                res = reduce<op, cdouble, cdouble>(in, dim, change_nan, nanval);
                break;
            case u32:
                res = reduce<op, uint, uint>(in, dim, change_nan, nanval);
                break;
            case s32:
                res = reduce<op, int, int>(in, dim, change_nan, nanval);
                break;
            case u64:
                res = reduce<op, uintl, uintl>(in, dim, change_nan, nanval);
                break;
            case s64:
                res = reduce<op, intl, intl>(in, dim, change_nan, nanval);
                break;
            case u16:
                res = reduce<op, ushort, uint>(in, dim, change_nan, nanval);
                break;
            case s16:
                res = reduce<op, short, int>(in, dim, change_nan, nanval);
                break;
            case u8:
                res = reduce<op, uchar, uint>(in, dim, change_nan, nanval);
                break;
            case b8: {
                if (op == af_mul_t) {
                    res = reduce<af_and_t, char, char>(in, dim, change_nan,
                                                       nanval);
                } else {
                    res = reduce<af_notzero_t, char, uint>(in, dim, change_nan,
                                                           nanval);
                }
            } break;
            case f16:
                res = reduce<op, half, float>(in, dim, change_nan, nanval);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_promote_by_key(af_array *keys_out, af_array *vals_out,
                                    const af_array keys, const af_array vals,
                                    const int dim, bool change_nan = false,
                                    double nanval = 0.0) {
    try {
        ARG_ASSERT(4, dim >= 0);
        ARG_ASSERT(4, dim < 4);

        const ArrayInfo &kinfo   = getInfo(keys);
        const ArrayInfo &in_info = getInfo(vals);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(2, kinfo.isVector());
        ARG_ASSERT(2, in_info.dims()[dim] == kinfo.dims()[0]);

        switch (type) {
            case f32:
                reduce_key<op, float, float>(keys_out, vals_out, keys, vals,
                                             dim, change_nan, nanval);
                break;
            case f64:
                reduce_key<op, double, double>(keys_out, vals_out, keys, vals,
                                               dim, change_nan, nanval);
                break;
            case c32:
                reduce_key<op, cfloat, cfloat>(keys_out, vals_out, keys, vals,
                                               dim, change_nan, nanval);
                break;
            case c64:
                reduce_key<op, cdouble, cdouble>(keys_out, vals_out, keys, vals,
                                                 dim, change_nan, nanval);
                break;
            case u32:
                reduce_key<op, uint, uint>(keys_out, vals_out, keys, vals, dim,
                                           change_nan, nanval);
                break;
            case s32:
                reduce_key<op, int, int>(keys_out, vals_out, keys, vals, dim,
                                         change_nan, nanval);
                break;
            case u64:
                reduce_key<op, uintl, uintl>(keys_out, vals_out, keys, vals,
                                             dim, change_nan, nanval);
                break;
            case s64:
                reduce_key<op, intl, intl>(keys_out, vals_out, keys, vals, dim,
                                           change_nan, nanval);
                break;
            case u16:
                reduce_key<op, ushort, uint>(keys_out, vals_out, keys, vals,
                                             dim, change_nan, nanval);
                break;
            case s16:
                reduce_key<op, short, int>(keys_out, vals_out, keys, vals, dim,
                                           change_nan, nanval);
                break;
            case u8:
                reduce_key<op, uchar, uint>(keys_out, vals_out, keys, vals, dim,
                                            change_nan, nanval);
                break;
            case b8:
                reduce_key<af_notzero_t, char, uint>(
                    keys_out, vals_out, keys, vals, dim, change_nan, nanval);
                break;
            case f16:
                reduce_key<op, half, float>(keys_out, vals_out, keys, vals, dim,
                                            change_nan, nanval);
                break;
            default: TYPE_ERROR(3, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_min(af_array *out, const af_array in, const int dim) {
    return reduce_common<af_min_t>(out, in, dim);
}

af_err af_max(af_array *out, const af_array in, const int dim) {
    return reduce_common<af_max_t>(out, in, dim);
}

af_err af_sum(af_array *out, const af_array in, const int dim) {
    return reduce_promote<af_add_t>(out, in, dim);
}

af_err af_product(af_array *out, const af_array in, const int dim) {
    return reduce_promote<af_mul_t>(out, in, dim);
}

af_err af_sum_nan(af_array *out, const af_array in, const int dim,
                  const double nanval) {
    return reduce_promote<af_add_t>(out, in, dim, true, nanval);
}

af_err af_product_nan(af_array *out, const af_array in, const int dim,
                      const double nanval) {
    return reduce_promote<af_mul_t>(out, in, dim, true, nanval);
}

af_err af_count(af_array *out, const af_array in, const int dim) {
    return reduce_type<af_notzero_t, uint>(out, in, dim);
}

af_err af_all_true(af_array *out, const af_array in, const int dim) {
    return reduce_type<af_and_t, char>(out, in, dim);
}

af_err af_any_true(af_array *out, const af_array in, const int dim) {
    return reduce_type<af_or_t, char>(out, in, dim);
}

// by key versions
af_err af_min_by_key(af_array *keys_out, af_array *vals_out,
                     const af_array keys, const af_array vals, const int dim) {
    return reduce_by_key_common<af_min_t>(keys_out, vals_out, keys, vals, dim);
}

af_err af_max_by_key(af_array *keys_out, af_array *vals_out,
                     const af_array keys, const af_array vals, const int dim) {
    return reduce_by_key_common<af_max_t>(keys_out, vals_out, keys, vals, dim);
}

af_err af_sum_by_key(af_array *keys_out, af_array *vals_out,
                     const af_array keys, const af_array vals, const int dim) {
    return reduce_promote_by_key<af_add_t>(keys_out, vals_out, keys, vals, dim);
}

af_err af_product_by_key(af_array *keys_out, af_array *vals_out,
                         const af_array keys, const af_array vals,
                         const int dim) {
    return reduce_promote_by_key<af_mul_t>(keys_out, vals_out, keys, vals, dim);
}

af_err af_sum_by_key_nan(af_array *keys_out, af_array *vals_out,
                         const af_array keys, const af_array vals,
                         const int dim, const double nanval) {
    return reduce_promote_by_key<af_add_t>(keys_out, vals_out, keys, vals, dim,
                                           true, nanval);
}

af_err af_product_by_key_nan(af_array *keys_out, af_array *vals_out,
                             const af_array keys, const af_array vals,
                             const int dim, const double nanval) {
    return reduce_promote_by_key<af_mul_t>(keys_out, vals_out, keys, vals, dim,
                                           true, nanval);
}

af_err af_count_by_key(af_array *keys_out, af_array *vals_out,
                       const af_array keys, const af_array vals,
                       const int dim) {
    return reduce_by_key_type<af_notzero_t, uint>(keys_out, vals_out, keys,
                                                  vals, dim);
}

af_err af_all_true_by_key(af_array *keys_out, af_array *vals_out,
                          const af_array keys, const af_array vals,
                          const int dim) {
    return reduce_by_key_type<af_and_t, char>(keys_out, vals_out, keys, vals,
                                              dim);
}

af_err af_any_true_by_key(af_array *keys_out, af_array *vals_out,
                          const af_array keys, const af_array vals,
                          const int dim) {
    return reduce_by_key_type<af_or_t, char>(keys_out, vals_out, keys, vals,
                                             dim);
}

template<af_op_t op, typename Ti, typename Tacc, typename Tret = double>
static inline Tret reduce_all(const af_array in, bool change_nan = false,
                              double nanval = 0) {
    return static_cast<Tret>(
        reduce_all<op, Ti, Tacc>(getArray<Ti>(in), change_nan, nanval));
}

template<af_op_t op, typename To>
static af_err reduce_all_type(double *real, double *imag, const af_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(0, real != nullptr);
        *real = 0;
        if (imag) { *imag = 0; }

        switch (type) {
            // clang-format off
            case f32: *real = reduce_all<op, float,   To>(in); break;
            case f64: *real = reduce_all<op, double,  To>(in); break;
            case c32: *real = reduce_all<op, cfloat,  To>(in); break;
            case c64: *real = reduce_all<op, cdouble, To>(in); break;
            case u32: *real = reduce_all<op, uint,    To>(in); break;
            case s32: *real = reduce_all<op, int,     To>(in); break;
            case u64: *real = reduce_all<op, uintl,   To>(in); break;
            case s64: *real = reduce_all<op, intl,    To>(in); break;
            case u16: *real = reduce_all<op, ushort,  To>(in); break;
            case s16: *real = reduce_all<op, short,   To>(in); break;
            case b8:  *real = reduce_all<op, char,    To>(in); break;
            case u8:  *real = reduce_all<op, uchar,   To>(in); break;
            case f16: *real = reduce_all<op, half,    To>(in); break;
            // clang-format on
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_all_common(double *real_val, double *imag_val,
                                const af_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(2, in_info.ndims() > 0);
        ARG_ASSERT(0, real_val != nullptr);
        *real_val = 0;
        if (imag_val != nullptr) { *imag_val = 0; }

        cfloat cfval;
        cdouble cdval;

        switch (type) {
            // clang-format off
            case f32: *real_val = reduce_all<op, float,  float>(in); break;
            case f64: *real_val = reduce_all<op, double, double>(in); break;
            case u32: *real_val = reduce_all<op, uint,   uint>(in); break;
            case s32: *real_val = reduce_all<op, int,    int>(in); break;
            case u64: *real_val = reduce_all<op, uintl,  uintl>(in); break;
            case s64: *real_val = reduce_all<op, intl,   intl>(in); break;
            case u16: *real_val = reduce_all<op, ushort, ushort>(in); break;
            case s16: *real_val = reduce_all<op, short,  short>(in); break;
            case b8:  *real_val = reduce_all<op, char,   char>(in); break;
            case u8:  *real_val = reduce_all<op, uchar,  uchar>(in); break;
            case f16: *real_val = reduce_all<op, half,   half>(in); break;
            // clang-format on
            case c32:
                cfval = reduce_all<op, cfloat, cfloat, cfloat>(in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cfval);
                *imag_val = imag(cfval);
                break;

            case c64:
                cdval = reduce_all<op, cdouble, cdouble, cdouble>(in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cdval);
                *imag_val = imag(cdval);
                break;

            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<af_op_t op>
static af_err reduce_all_promote(double *real_val, double *imag_val,
                                 const af_array in, bool change_nan = false,
                                 double nanval = 0) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(0, real_val != nullptr);
        *real_val = 0;
        if (imag_val) { *imag_val = 0; }

        cfloat cfval;
        cdouble cdval;

        switch (type) {
            // clang-format off
            case f32: *real_val = reduce_all<op, float,   float>(in, change_nan, nanval); break;
            case f64: *real_val = reduce_all<op, double, double>(in, change_nan, nanval); break;
            case u32: *real_val = reduce_all<op, uint,     uint>(in, change_nan, nanval); break;
            case s32: *real_val = reduce_all<op, int,       int>(in, change_nan, nanval); break;
            case u64: *real_val = reduce_all<op, uintl,   uintl>(in, change_nan, nanval); break;
            case s64: *real_val = reduce_all<op, intl,     intl>(in, change_nan, nanval); break;
            case u16: *real_val = reduce_all<op, ushort,   uint>(in, change_nan, nanval); break;
            case s16: *real_val = reduce_all<op, short,     int>(in, change_nan, nanval); break;
            case u8:  *real_val = reduce_all<op, uchar,    uint>(in, change_nan, nanval); break;
            // clang-format on
            case b8: {
                if (op == af_mul_t) {
                    *real_val = reduce_all<af_and_t, char, char>(in, change_nan,
                                                                 nanval);
                } else {
                    *real_val = reduce_all<af_notzero_t, char, uint>(
                        in, change_nan, nanval);
                }
            } break;
            case c32:
                cfval = reduce_all<op, cfloat, cfloat, cfloat>(in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cfval);
                *imag_val = imag(cfval);
                break;

            case c64:
                cdval = reduce_all<op, cdouble, cdouble, cdouble>(in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cdval);
                *imag_val = imag(cdval);
                break;
            case f16:
                *real_val = reduce_all<op, half, float>(in, change_nan, nanval);
                break;

            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_min_all(double *real, double *imag, const af_array in) {
    return reduce_all_common<af_min_t>(real, imag, in);
}

af_err af_max_all(double *real, double *imag, const af_array in) {
    return reduce_all_common<af_max_t>(real, imag, in);
}

af_err af_sum_all(double *real, double *imag, const af_array in) {
    return reduce_all_promote<af_add_t>(real, imag, in);
}

af_err af_product_all(double *real, double *imag, const af_array in) {
    return reduce_all_promote<af_mul_t>(real, imag, in);
}

af_err af_count_all(double *real, double *imag, const af_array in) {
    return reduce_all_type<af_notzero_t, uint>(real, imag, in);
}

af_err af_all_true_all(double *real, double *imag, const af_array in) {
    return reduce_all_type<af_and_t, char>(real, imag, in);
}

af_err af_any_true_all(double *real, double *imag, const af_array in) {
    return reduce_all_type<af_or_t, char>(real, imag, in);
}

template<af_op_t op, typename T>
static inline void ireduce(af_array *res, af_array *loc, const af_array in,
                           const int dim) {
    const Array<T> In = getArray<T>(in);
    dim4 odims        = In.dims();
    odims[dim]        = 1;

    Array<T> Res    = createEmptyArray<T>(odims);
    Array<uint> Loc = createEmptyArray<uint>(odims);
    ireduce<op, T>(Res, Loc, In, dim);

    *res = getHandle(Res);
    *loc = getHandle(Loc);
}

template<af_op_t op, typename T>
static inline void rreduce(af_array *res, af_array *loc, const af_array in,
                           const int dim, const af_array ragged_len) {
    const Array<T> In     = getArray<T>(in);
    const Array<uint> Len = getArray<uint>(ragged_len);
    dim4 odims            = In.dims();
    odims[dim]            = 1;

    Array<T> Res    = createEmptyArray<T>(odims);
    Array<uint> Loc = createEmptyArray<uint>(odims);
    rreduce<op, T>(Res, Loc, In, dim, Len);

    *res = getHandle(Res);
    *loc = getHandle(Loc);
}

template<af_op_t op>
static af_err ireduce_common(af_array *val, af_array *idx, const af_array in,
                             const int dim) {
    try {
        ARG_ASSERT(3, dim >= 0);
        ARG_ASSERT(3, dim < 4);

        const ArrayInfo &in_info = getInfo(in);
        ARG_ASSERT(2, in_info.ndims() > 0);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *val = retain(in);
            *idx = createHandleFromValue<uint>(in_info.dims(), 0);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res, loc;

        switch (type) {
            case f32: ireduce<op, float>(&res, &loc, in, dim); break;
            case f64: ireduce<op, double>(&res, &loc, in, dim); break;
            case c32: ireduce<op, cfloat>(&res, &loc, in, dim); break;
            case c64: ireduce<op, cdouble>(&res, &loc, in, dim); break;
            case u32: ireduce<op, uint>(&res, &loc, in, dim); break;
            case s32: ireduce<op, int>(&res, &loc, in, dim); break;
            case u64: ireduce<op, uintl>(&res, &loc, in, dim); break;
            case s64: ireduce<op, intl>(&res, &loc, in, dim); break;
            case u16: ireduce<op, ushort>(&res, &loc, in, dim); break;
            case s16: ireduce<op, short>(&res, &loc, in, dim); break;
            case b8: ireduce<op, char>(&res, &loc, in, dim); break;
            case u8: ireduce<op, uchar>(&res, &loc, in, dim); break;
            case f16: ireduce<op, half>(&res, &loc, in, dim); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*val, res);
        std::swap(*idx, loc);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_imin(af_array *val, af_array *idx, const af_array in, const int dim) {
    return ireduce_common<af_min_t>(val, idx, in, dim);
}

af_err af_imax(af_array *val, af_array *idx, const af_array in, const int dim) {
    return ireduce_common<af_max_t>(val, idx, in, dim);
}

template<af_op_t op>
static af_err rreduce_common(af_array *val, af_array *idx, const af_array in,
                             const af_array ragged_len, const int dim) {
    try {
        ARG_ASSERT(3, dim >= 0);
        ARG_ASSERT(3, dim < 4);

        const ArrayInfo &in_info = getInfo(in);
        ARG_ASSERT(2, in_info.ndims() > 0);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *val = retain(in);
            *idx = createHandleFromValue<uint>(in_info.dims(), 0);
            return AF_SUCCESS;
        }

        // Make sure ragged_len.dims == in.dims(), except on reduced dim
        const ArrayInfo &ragged_info = getInfo(ragged_len);
        dim4 test_dim                = in_info.dims();
        test_dim[dim]                = 1;
        ARG_ASSERT(4, test_dim == ragged_info.dims());

        af_dtype keytype = ragged_info.getType();
        if (keytype != u32) { TYPE_ERROR(4, keytype); }

        af_dtype type = in_info.getType();
        af_array res, loc;

        switch (type) {
            case f32:
                rreduce<op, float>(&res, &loc, in, dim, ragged_len);
                break;
            case f64:
                rreduce<op, double>(&res, &loc, in, dim, ragged_len);
                break;
            case c32:
                rreduce<op, cfloat>(&res, &loc, in, dim, ragged_len);
                break;
            case c64:
                rreduce<op, cdouble>(&res, &loc, in, dim, ragged_len);
                break;
            case u32: rreduce<op, uint>(&res, &loc, in, dim, ragged_len); break;
            case s32: rreduce<op, int>(&res, &loc, in, dim, ragged_len); break;
            case u64:
                rreduce<op, uintl>(&res, &loc, in, dim, ragged_len);
                break;
            case s64: rreduce<op, intl>(&res, &loc, in, dim, ragged_len); break;
            case u16:
                rreduce<op, ushort>(&res, &loc, in, dim, ragged_len);
                break;
            case s16:
                rreduce<op, short>(&res, &loc, in, dim, ragged_len);
                break;
            case b8: rreduce<op, char>(&res, &loc, in, dim, ragged_len); break;
            case u8: rreduce<op, uchar>(&res, &loc, in, dim, ragged_len); break;
            case f16: rreduce<op, half>(&res, &loc, in, dim, ragged_len); break;
            default: TYPE_ERROR(2, type);
        }

        std::swap(*val, res);
        std::swap(*idx, loc);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_max_ragged(af_array *val, af_array *idx, const af_array in,
                     const af_array ragged_len, const int dim) {
    return rreduce_common<af_max_t>(val, idx, in, ragged_len, dim);
}

template<af_op_t op, typename T, typename Tret = T>
static inline Tret ireduce_all(unsigned *loc, const af_array in) {
    return static_cast<Tret>(ireduce_all<op, T>(loc, getArray<T>(in)));
}

template<af_op_t op>
static af_err ireduce_all_common(double *real_val, double *imag_val,
                                 unsigned *loc, const af_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype type            = in_info.getType();

        ARG_ASSERT(3, in_info.ndims() > 0);
        ARG_ASSERT(0, real_val != nullptr);
        *real_val = 0;
        if (imag_val) { *imag_val = 0; }

        cfloat cfval;
        cdouble cdval;

        switch (type) {
            case f32:
                *real_val = ireduce_all<op, float, double>(loc, in);
                break;
            case f64:
                *real_val = ireduce_all<op, double, double>(loc, in);
                break;
            case u32: *real_val = ireduce_all<op, uint, double>(loc, in); break;
            case s32: *real_val = ireduce_all<op, int, double>(loc, in); break;
            case u64:
                *real_val = ireduce_all<op, uintl, double>(loc, in);
                break;
            case s64: *real_val = ireduce_all<op, intl, double>(loc, in); break;
            case u16:
                *real_val = ireduce_all<op, ushort, double>(loc, in);
                break;
            case s16:
                *real_val = ireduce_all<op, short, double>(loc, in);
                break;
            case b8: *real_val = ireduce_all<op, char, double>(loc, in); break;
            case u8: *real_val = ireduce_all<op, uchar, double>(loc, in); break;

            case c32:
                cfval = ireduce_all<op, cfloat>(loc, in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cfval);
                *imag_val = imag(cfval);
                break;

            case c64:
                cdval = ireduce_all<op, cdouble>(loc, in);
                ARG_ASSERT(1, imag_val != nullptr);
                *real_val = real(cdval);
                *imag_val = imag(cdval);
                break;

            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_imin_all(double *real, double *imag, unsigned *idx,
                   const af_array in) {
    return ireduce_all_common<af_min_t>(real, imag, idx, in);
}

af_err af_imax_all(double *real, double *imag, unsigned *idx,
                   const af_array in) {
    return ireduce_all_common<af_max_t>(real, imag, idx, in);
}

af_err af_sum_nan_all(double *real, double *imag, const af_array in,
                      const double nanval) {
    return reduce_all_promote<af_add_t>(real, imag, in, true, nanval);
}

af_err af_product_nan_all(double *real, double *imag, const af_array in,
                          const double nanval) {
    return reduce_all_promote<af_mul_t>(real, imag, in, true, nanval);
}
