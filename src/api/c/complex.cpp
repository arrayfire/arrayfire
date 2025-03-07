/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <optypes.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>

#include <complex.hpp>

using af::dim4;
using arrayfire::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::conj;
using detail::imag;
using detail::real;

template<typename To, typename Ti>
static inline af_array cplx(const af_array lhs, const af_array rhs,
                            const dim4 &odims) {
    af_array res =
        getHandle(cplx<To, Ti>(castArray<Ti>(lhs), castArray<Ti>(rhs), odims));
    return res;
}

af_err af_cplx2(af_array *out, const af_array lhs, const af_array rhs,
                bool batchMode) {
    try {
        af_dtype type = implicit(lhs, rhs);

        if (type == c32 || type == c64) {
            AF_ERROR("Inputs to cplx2 can not be of complex type", AF_ERR_ARG);
        }

        if (type != f64) { type = f32; }
        dim4 odims =
            getOutDims(getInfo(lhs).dims(), getInfo(rhs).dims(), batchMode);
        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res;
        switch (type) {
            case f32: res = cplx<cfloat, float>(lhs, rhs, odims); break;
            case f64: res = cplx<cdouble, double>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_cplx(af_array *out, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (type == c32 || type == c64) {
            AF_ERROR("Inputs to cplx2 can not be of complex type", AF_ERR_ARG);
        }
        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array tmp;
        AF_CHECK(af_constant(&tmp, 0, info.ndims(), info.dims().get(), type));

        af_array res;
        switch (type) {
            case f32: res = cplx<cfloat, float>(in, tmp, info.dims()); break;
            case f64: res = cplx<cdouble, double>(in, tmp, info.dims()); break;

            default: TYPE_ERROR(0, type);
        }

        AF_CHECK(af_release_array(tmp));

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_real(af_array *out, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (type != c32 && type != c64) { return af_retain_array(out, in); }
        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array res;
        switch (type) {
            case c32:
                res = getHandle(real<float, cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(real<double, cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_imag(af_array *out, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (type != c32 && type != c64) {
            return af_constant(out, 0, info.ndims(), info.dims().get(), type);
        }
        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array res;
        switch (type) {
            case c32:
                res = getHandle(imag<float, cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(imag<double, cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_conjg(af_array *out, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();

        if (type != c32 && type != c64) { return af_retain_array(out, in); }
        if (info.ndims() == 0) { return af_retain_array(out, in); }

        af_array res;
        switch (type) {
            case c32:
                res = getHandle(conj<cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(conj<cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_abs(af_array *out, const af_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        af_dtype in_type         = in_info.getType();
        af_array res;

        // Convert all inputs to floats / doubles
        af_dtype type = implicit(in_type, f32);
        if (in_type == f16) { type = f16; }
        if (in_info.ndims() == 0) { return af_retain_array(out, in); }

        switch (type) {
            // clang-format off
            case f32: res = getHandle(detail::abs<float, float>(castArray<float>(in))); break;
            case f64: res = getHandle(detail::abs<double, double>(castArray<double>(in))); break;
            case c32: res = getHandle(detail::abs<float, cfloat>(castArray<cfloat>(in))); break;
            case c64: res = getHandle(detail::abs<double, cdouble>(castArray<cdouble>(in))); break;
            case f16: res = getHandle(detail::abs<half, half>(getArray<half>(in))); break;
            // clang-format on
            default: TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}
