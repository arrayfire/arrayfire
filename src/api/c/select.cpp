/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <af/data.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <select.hpp>

using namespace detail;
using af::dim4;

template<typename T>
af_array select(const af_array cond, const af_array a, const af_array b, const dim4 &odims)
{
    Array<T> out = createEmptyArray<T>(odims);
    select(out, getArray<char>(cond), getArray<T>(a), getArray<T>(b));
    return getHandle<T>(out);
}

af_err af_select(af_array *out, const af_array cond, const af_array a, const af_array b)
{
    try {
        ArrayInfo ainfo = getInfo(a);
        ArrayInfo binfo = getInfo(b);
        ArrayInfo cinfo = getInfo(cond);

        ARG_ASSERT(2, ainfo.getType() == binfo.getType());
        ARG_ASSERT(1, cinfo.getType() == b8);

        DIM_ASSERT(1, cinfo.ndims() == std::min(ainfo.ndims(), binfo.ndims()));

        dim4 adims = ainfo.dims();
        dim4 bdims = binfo.dims();
        dim4 cdims = cinfo.dims();
        dim4 odims(1, 1, 1, 1);

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == std::min(adims[i], bdims[i]));
            DIM_ASSERT(2, adims[i] == bdims[i] || adims[i] == 1 || bdims[i] == 1);
            odims[i] = std::max(adims[i], bdims[i]);
        }

        af_array res;

        switch (ainfo.getType()) {
        case f32: res = select<float  >(cond, a, b, odims); break;
        case f64: res = select<double >(cond, a, b, odims); break;
        case c32: res = select<cfloat >(cond, a, b, odims); break;
        case c64: res = select<cdouble>(cond, a, b, odims); break;
        case s32: res = select<int    >(cond, a, b, odims); break;
        case u32: res = select<uint   >(cond, a, b, odims); break;
        case s64: res = select<intl   >(cond, a, b, odims); break;
        case u64: res = select<uintl  >(cond, a, b, odims); break;
        case u8:  res = select<uchar  >(cond, a, b, odims); break;
        case b8:  res = select<char   >(cond, a, b, odims); break;
        default:  TYPE_ERROR(2, ainfo.getType());
        }

        std::swap(*out, res);
    } CATCHALL;
    return AF_SUCCESS;
}

template<typename T, bool flip>
af_array select_scalar(const af_array cond, const af_array a, const double b, const dim4 &odims)
{
    Array<T> out = createEmptyArray<T>(odims);
    select_scalar<T, flip>(out, getArray<char>(cond), getArray<T>(a), b);
    return getHandle<T>(out);
}

af_err af_select_scalar_r(af_array *out, const af_array cond, const af_array a, const double b)
{
    try {
        ArrayInfo ainfo = getInfo(a);
        ArrayInfo cinfo = getInfo(cond);

        ARG_ASSERT(1, cinfo.getType() == b8);
        DIM_ASSERT(1, cinfo.ndims() == ainfo.ndims());

        dim4 adims = ainfo.dims();
        dim4 cdims = cinfo.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == adims[i]);
        }

        af_array res;

        switch (ainfo.getType()) {
        case f32: res = select_scalar<float  , false>(cond, a, b, adims); break;
        case f64: res = select_scalar<double , false>(cond, a, b, adims); break;
        case c32: res = select_scalar<cfloat , false>(cond, a, b, adims); break;
        case c64: res = select_scalar<cdouble, false>(cond, a, b, adims); break;
        case s32: res = select_scalar<int    , false>(cond, a, b, adims); break;
        case u32: res = select_scalar<uint   , false>(cond, a, b, adims); break;
        case s64: res = select_scalar<intl   , false>(cond, a, b, adims); break;
        case u64: res = select_scalar<uintl  , false>(cond, a, b, adims); break;
        case u8:  res = select_scalar<uchar  , false>(cond, a, b, adims); break;
        case b8:  res = select_scalar<char   , false>(cond, a, b, adims); break;
        default:  TYPE_ERROR(2, ainfo.getType());
        }

        std::swap(*out, res);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_select_scalar_l(af_array *out, const af_array cond, const double a, const af_array b)
{
    try {
        ArrayInfo binfo = getInfo(b);
        ArrayInfo cinfo = getInfo(cond);

        ARG_ASSERT(1, cinfo.getType() == b8);
        DIM_ASSERT(1, cinfo.ndims() == binfo.ndims());

        dim4 bdims = binfo.dims();
        dim4 cdims = cinfo.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == bdims[i]);
        }

        af_array res;

        switch (binfo.getType()) {
        case f32: res = select_scalar<float  , true >(cond, b, a, bdims); break;
        case f64: res = select_scalar<double , true >(cond, b, a, bdims); break;
        case c32: res = select_scalar<cfloat , true >(cond, b, a, bdims); break;
        case c64: res = select_scalar<cdouble, true >(cond, b, a, bdims); break;
        case s32: res = select_scalar<int    , true >(cond, b, a, bdims); break;
        case u32: res = select_scalar<uint   , true >(cond, b, a, bdims); break;
        case s64: res = select_scalar<intl   , true >(cond, b, a, bdims); break;
        case u64: res = select_scalar<uintl  , true >(cond, b, a, bdims); break;
        case u8:  res = select_scalar<uchar  , true >(cond, b, a, bdims); break;
        case b8:  res = select_scalar<char   , true >(cond, b, a, bdims); break;
        default:  TYPE_ERROR(2, binfo.getType());
        }

        std::swap(*out, res);
    } CATCHALL;
    return AF_SUCCESS;
}
