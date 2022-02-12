/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <select.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>

using af::dim4;
using common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createSelectNode;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
af_array select(const af_array cond, const af_array a, const af_array b,
                const dim4& odims) {
    Array<T> out = createSelectNode(getArray<char>(cond), getArray<T>(a),
                                    getArray<T>(b), odims);
    return getHandle<T>(out);
}

af_err af_select(af_array* out, const af_array cond, const af_array a,
                 const af_array b) {
    try {
        const ArrayInfo& ainfo     = getInfo(a);
        const ArrayInfo& binfo     = getInfo(b);
        const ArrayInfo& cond_info = getInfo(cond);

        if (cond_info.ndims() == 0) { return af_retain_array(out, cond); }

        ARG_ASSERT(2, ainfo.getType() == binfo.getType());
        ARG_ASSERT(1, cond_info.getType() == b8);

        dim4 adims     = ainfo.dims();
        dim4 bdims     = binfo.dims();
        dim4 cond_dims = cond_info.dims();
        dim4 odims(1, 1, 1, 1);

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(2, (adims[i] == bdims[i] && adims[i] == cond_dims[i]) ||
                              adims[i] == 1 || bdims[i] == 1 ||
                              cond_dims[i] == 1);
            odims[i] = std::max(std::max(adims[i], bdims[i]), cond_dims[i]);
        }

        af_array res;

        switch (ainfo.getType()) {
            case f32: res = select<float>(cond, a, b, odims); break;
            case f64: res = select<double>(cond, a, b, odims); break;
            case c32: res = select<cfloat>(cond, a, b, odims); break;
            case c64: res = select<cdouble>(cond, a, b, odims); break;
            case s32: res = select<int>(cond, a, b, odims); break;
            case u32: res = select<uint>(cond, a, b, odims); break;
            case s64: res = select<intl>(cond, a, b, odims); break;
            case u64: res = select<uintl>(cond, a, b, odims); break;
            case s16: res = select<short>(cond, a, b, odims); break;
            case u16: res = select<ushort>(cond, a, b, odims); break;
            case u8: res = select<uchar>(cond, a, b, odims); break;
            case b8: res = select<char>(cond, a, b, odims); break;
            case f16: res = select<half>(cond, a, b, odims); break;
            default: TYPE_ERROR(2, ainfo.getType());
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename ArrayType, typename ScalarType, bool flip>
af_array select_scalar(const af_array cond, const af_array a,
                       const ScalarType b, const dim4& odims) {
    auto scalar = detail::scalar<ArrayType>(b);
    auto out    = createSelectNode<ArrayType, flip>(
        getArray<char>(cond), getArray<ArrayType>(a), scalar, odims);
    return getHandle(out);
}

template<typename ScalarType, bool IsScalarTrueOutput>
af_err selectScalar(af_array* out, const af_array cond, const af_array e,
                    const ScalarType c) {
    try {
        const ArrayInfo& einfo = getInfo(e);
        const ArrayInfo& cinfo = getInfo(cond);

        ARG_ASSERT(1, cinfo.getType() == b8);

        dim4 edims     = einfo.dims();
        dim4 cond_dims = cinfo.dims();
        dim4 odims(1);

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cond_dims[i] == edims[i] || cond_dims[i] == 1 ||
                              edims[i] == 1);
            odims[i] = std::max(cond_dims[i], edims[i]);
        }

        af_array res;

        switch (einfo.getType()) {
            case f16:
                res = select_scalar<half, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case f32:
                res = select_scalar<float, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case f64:
                res = select_scalar<double, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case c32:
                res = select_scalar<cfloat, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case c64:
                res = select_scalar<cdouble, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s32:
                res = select_scalar<int, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u32:
                res = select_scalar<uint, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s16:
                res = select_scalar<short, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u16:
                res = select_scalar<ushort, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s64:
                res = select_scalar<intl, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u64:
                res = select_scalar<uintl, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u8:
                res = select_scalar<uchar, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case b8:
                res = select_scalar<char, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            default: TYPE_ERROR((IsScalarTrueOutput ? 3 : 2), einfo.getType());
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_select_scalar_r(af_array* out, const af_array cond, const af_array a,
                          const double b) {
    return selectScalar<double, false>(out, cond, a, b);
}

af_err af_select_scalar_r_long(af_array* out, const af_array cond,
                               const af_array a, const long long b) {
    return selectScalar<long long, false>(out, cond, a, b);
}

af_err af_select_scalar_r_ulong(af_array* out, const af_array cond,
                                const af_array a, const unsigned long long b) {
    return selectScalar<unsigned long long, false>(out, cond, a, b);
}

af_err af_select_scalar_l(af_array* out, const af_array cond, const double a,
                          const af_array b) {
    return selectScalar<double, true>(out, cond, b, a);
}

af_err af_select_scalar_l_long(af_array* out, const af_array cond,
                               const long long a, const af_array b) {
    return selectScalar<long long, true>(out, cond, b, a);
}

af_err af_select_scalar_l_ulong(af_array* out, const af_array cond,
                                const unsigned long long a, const af_array b) {
    return selectScalar<unsigned long long, true>(out, cond, b, a);
}
