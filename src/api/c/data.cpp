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
#include <copy.hpp>
#include <diagonal.hpp>
#include <handle.hpp>
#include <identity.hpp>
#include <iota.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <range.hpp>
#include <triangle.hpp>
#include <af/array.h>
#include <af/data.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/util.h>
#include <complex>

using af::dim4;
using namespace detail;

dim4 verifyDims(const unsigned ndims, const dim_t *const dims) {
    DIM_ASSERT(1, ndims >= 1);

    dim4 d(1, 1, 1, 1);

    for (unsigned i = 0; i < ndims; i++) {
        d[i] = dims[i];
        DIM_ASSERT(2, dims[i] >= 1);
    }

    return d;
}

// Strong Exception Guarantee
af_err af_constant(af_array *result, const double value, const unsigned ndims,
                   const dim_t *const dims, const af_dtype type) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d(1, 1, 1, 1);
        if (ndims <= 0) {
            return af_create_handle(result, 0, nullptr, type);
        } else {
            d = verifyDims(ndims, dims);
        }

        switch (type) {
            case f32: out = createHandleFromValue<float>(d, value); break;
            case c32: out = createHandleFromValue<cfloat>(d, value); break;
            case f64: out = createHandleFromValue<double>(d, value); break;
            case c64: out = createHandleFromValue<cdouble>(d, value); break;
            case b8: out = createHandleFromValue<char>(d, value); break;
            case s32: out = createHandleFromValue<int>(d, value); break;
            case u32: out = createHandleFromValue<uint>(d, value); break;
            case u8: out = createHandleFromValue<uchar>(d, value); break;
            case s64: out = createHandleFromValue<intl>(d, value); break;
            case u64: out = createHandleFromValue<uintl>(d, value); break;
            case s16: out = createHandleFromValue<short>(d, value); break;
            case u16: out = createHandleFromValue<ushort>(d, value); break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename To, typename Ti>
static inline af_array createCplx(dim4 dims, const Ti real, const Ti imag) {
    To cval      = scalar<To, Ti>(real, imag);
    af_array out = getHandle(createValueArray<To>(dims, cval));
    return out;
}

af_err af_constant_complex(af_array *result, const double real,
                           const double imag, const unsigned ndims,
                           const dim_t *const dims, af_dtype type) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d(1, 1, 1, 1);
        if (ndims <= 0) {
            return af_create_handle(result, 0, nullptr, type);
        } else {
            d = verifyDims(ndims, dims);
        }

        switch (type) {
            case c32: out = createCplx<cfloat, float>(d, real, imag); break;
            case c64: out = createCplx<cdouble, double>(d, real, imag); break;
            default: TYPE_ERROR(5, type);
        }

        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_constant_long(af_array *result, const intl val, const unsigned ndims,
                        const dim_t *const dims) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d(1, 1, 1, 1);
        if (ndims <= 0) {
            return af_create_handle(result, 0, nullptr, s64);
        } else {
            d = verifyDims(ndims, dims);
        }

        out = getHandle(createValueArray<intl>(d, val));

        std::swap(*result, out);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_constant_ulong(af_array *result, const uintl val,
                         const unsigned ndims, const dim_t *const dims) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d(1, 1, 1, 1);
        if (ndims <= 0) {
            return af_create_handle(result, 0, nullptr, u64);
        } else {
            d = verifyDims(ndims, dims);
        }
        out = getHandle(createValueArray<uintl>(d, val));

        std::swap(*result, out);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline af_array identity_(const af::dim4 &dims) {
    return getHandle(detail::identity<T>(dims));
}

af_err af_identity(af_array *out, const unsigned ndims, const dim_t *const dims,
                   const af_dtype type) {
    try {
        af_array result;
        AF_CHECK(af_init());

        if (ndims == 0) { return af_create_handle(out, 0, nullptr, type); }

        dim4 d = verifyDims(ndims, dims);

        switch (type) {
            case f32: result = identity_<float>(d); break;
            case c32: result = identity_<cfloat>(d); break;
            case f64: result = identity_<double>(d); break;
            case c64: result = identity_<cdouble>(d); break;
            case s32: result = identity_<int>(d); break;
            case u32: result = identity_<uint>(d); break;
            case u8: result = identity_<uchar>(d); break;
            case u64: result = identity_<uintl>(d); break;
            case s64: result = identity_<intl>(d); break;
            case u16: result = identity_<ushort>(d); break;
            case s16:
                result = identity_<short>(d);
                break;
                // Removed because of bool type. Functions implementations
                // exist.
            case b8: result = identity_<char>(d); break;
            default: TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline af_array range_(const dim4 &d, const int seq_dim) {
    return getHandle(range<T>(d, seq_dim));
}

// Strong Exception Guarantee
af_err af_range(af_array *result, const unsigned ndims, const dim_t *const dims,
                const int seq_dim, const af_dtype type) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d(0);
        if (ndims <= 0) {
            return af_create_handle(result, 0, nullptr, type);
        } else {
            d = verifyDims(ndims, dims);
        }

        switch (type) {
            case f32: out = range_<float>(d, seq_dim); break;
            case f64: out = range_<double>(d, seq_dim); break;
            case s32: out = range_<int>(d, seq_dim); break;
            case u32: out = range_<uint>(d, seq_dim); break;
            case s64: out = range_<intl>(d, seq_dim); break;
            case u64: out = range_<uintl>(d, seq_dim); break;
            case s16: out = range_<short>(d, seq_dim); break;
            case u16: out = range_<ushort>(d, seq_dim); break;
            case u8: out = range_<uchar>(d, seq_dim); break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline af_array iota_(const dim4 &dims, const dim4 &tile_dims) {
    return getHandle(iota<T>(dims, tile_dims));
}

// Strong Exception Guarantee
af_err af_iota(af_array *result, const unsigned ndims, const dim_t *const dims,
               const unsigned t_ndims, const dim_t *const tdims,
               const af_dtype type) {
    try {
        af_array out;
        AF_CHECK(af_init());

        if (ndims == 0) { return af_create_handle(result, 0, nullptr, type); }

        DIM_ASSERT(1, ndims > 0 && ndims <= 4);
        DIM_ASSERT(3, t_ndims > 0 && t_ndims <= 4);

        dim4 d = verifyDims(ndims, dims);
        dim4 t = verifyDims(t_ndims, tdims);

        switch (type) {
            case f32: out = iota_<float>(d, t); break;
            case f64: out = iota_<double>(d, t); break;
            case s32: out = iota_<int>(d, t); break;
            case u32: out = iota_<uint>(d, t); break;
            case s64: out = iota_<intl>(d, t); break;
            case u64: out = iota_<uintl>(d, t); break;
            case s16: out = iota_<short>(d, t); break;
            case u16: out = iota_<ushort>(d, t); break;
            case u8: out = iota_<uchar>(d, t); break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline af_array diagCreate(const af_array in, const int num) {
    return getHandle(diagCreate<T>(getArray<T>(in), num));
}

template<typename T>
static inline af_array diagExtract(const af_array in, const int num) {
    return getHandle(diagExtract<T>(getArray<T>(in), num));
}

af_err af_diag_create(af_array *out, const af_array in, const int num) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        DIM_ASSERT(1, in_info.ndims() <= 2);
        af_dtype type = in_info.getType();

        af_array result;

        if (in_info.dims()[0] == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        switch (type) {
            case f32: result = diagCreate<float>(in, num); break;
            case c32: result = diagCreate<cfloat>(in, num); break;
            case f64: result = diagCreate<double>(in, num); break;
            case c64: result = diagCreate<cdouble>(in, num); break;
            case s32: result = diagCreate<int>(in, num); break;
            case u32: result = diagCreate<uint>(in, num); break;
            case s64: result = diagCreate<intl>(in, num); break;
            case u64: result = diagCreate<uintl>(in, num); break;
            case s16: result = diagCreate<short>(in, num); break;
            case u16: result = diagCreate<ushort>(in, num); break;
            case u8:
                result = diagCreate<uchar>(in, num);
                break;
                // Removed because of bool type. Functions implementations
                // exist.
            case b8: result = diagCreate<char>(in, num); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, result);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_diag_extract(af_array *out, const af_array in, const int num) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype type            = in_info.getType();

        if (in_info.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        DIM_ASSERT(1, in_info.ndims() >= 2);

        af_array result;
        switch (type) {
            case f32: result = diagExtract<float>(in, num); break;
            case c32: result = diagExtract<cfloat>(in, num); break;
            case f64: result = diagExtract<double>(in, num); break;
            case c64: result = diagExtract<cdouble>(in, num); break;
            case s32: result = diagExtract<int>(in, num); break;
            case u32: result = diagExtract<uint>(in, num); break;
            case s64: result = diagExtract<intl>(in, num); break;
            case u64: result = diagExtract<uintl>(in, num); break;
            case s16: result = diagExtract<short>(in, num); break;
            case u16: result = diagExtract<ushort>(in, num); break;
            case u8:
                result = diagExtract<uchar>(in, num);
                break;
                // Removed because of bool type. Functions implementations
                // exist.
            case b8: result = diagExtract<char>(in, num); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, result);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T, bool is_upper>
af_array triangle(const af_array in, bool is_unit_diag) {
    if (is_unit_diag)
        return getHandle(triangle<T, is_upper, true>(getArray<T>(in)));
    else
        return getHandle(triangle<T, is_upper, false>(getArray<T>(in)));
}

af_err af_lower(af_array *out, const af_array in, bool is_unit_diag) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array res;
        switch (type) {
            case f32: res = triangle<float, false>(in, is_unit_diag); break;
            case f64: res = triangle<double, false>(in, is_unit_diag); break;
            case c32: res = triangle<cfloat, false>(in, is_unit_diag); break;
            case c64: res = triangle<cdouble, false>(in, is_unit_diag); break;
            case s32: res = triangle<int, false>(in, is_unit_diag); break;
            case u32: res = triangle<uint, false>(in, is_unit_diag); break;
            case s64: res = triangle<intl, false>(in, is_unit_diag); break;
            case u64: res = triangle<uintl, false>(in, is_unit_diag); break;
            case s16: res = triangle<short, false>(in, is_unit_diag); break;
            case u16: res = triangle<ushort, false>(in, is_unit_diag); break;
            case u8: res = triangle<uchar, false>(in, is_unit_diag); break;
            case b8: res = triangle<char, false>(in, is_unit_diag); break;
        }
        std::swap(*out, res);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_upper(af_array *out, const af_array in, bool is_unit_diag) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array res;
        switch (type) {
            case f32: res = triangle<float, true>(in, is_unit_diag); break;
            case f64: res = triangle<double, true>(in, is_unit_diag); break;
            case c32: res = triangle<cfloat, true>(in, is_unit_diag); break;
            case c64: res = triangle<cdouble, true>(in, is_unit_diag); break;
            case s32: res = triangle<int, true>(in, is_unit_diag); break;
            case u32: res = triangle<uint, true>(in, is_unit_diag); break;
            case s64: res = triangle<intl, true>(in, is_unit_diag); break;
            case u64: res = triangle<uintl, true>(in, is_unit_diag); break;
            case s16: res = triangle<short, true>(in, is_unit_diag); break;
            case u16: res = triangle<ushort, true>(in, is_unit_diag); break;
            case u8: res = triangle<uchar, true>(in, is_unit_diag); break;
            case b8: res = triangle<char, true>(in, is_unit_diag); break;
        }
        std::swap(*out, res);
    }
    CATCHALL
    return AF_SUCCESS;
}
