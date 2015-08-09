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
void replace(af_array a, const af_array cond, const af_array b)
{
    select(getWritableArray<T>(a), getArray<char>(cond), getArray<T>(a), getArray<T>(b));
}

af_err af_replace(af_array a, const af_array cond, const af_array b)
{
    try {
        ArrayInfo ainfo = getInfo(a);
        ArrayInfo binfo = getInfo(b);
        ArrayInfo cinfo = getInfo(cond);

        ARG_ASSERT(2, ainfo.getType() == binfo.getType());
        ARG_ASSERT(1, cinfo.getType() == b8);

        DIM_ASSERT(1, ainfo.ndims() >= binfo.ndims());
        DIM_ASSERT(1, cinfo.ndims() == std::min(ainfo.ndims(), binfo.ndims()));

        dim4 adims = ainfo.dims();
        dim4 bdims = binfo.dims();
        dim4 cdims = cinfo.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == std::min(adims[i], bdims[i]));
            DIM_ASSERT(2, adims[i] == bdims[i] || bdims[i] == 1);
        }

        switch (ainfo.getType()) {
        case f32: replace<float  >(a, cond, b); break;
        case f64: replace<double >(a, cond, b); break;
        case c32: replace<cfloat >(a, cond, b); break;
        case c64: replace<cdouble>(a, cond, b); break;
        case s32: replace<int    >(a, cond, b); break;
        case u32: replace<uint   >(a, cond, b); break;
        case s64: replace<intl   >(a, cond, b); break;
        case u64: replace<uintl  >(a, cond, b); break;
        case u8:  replace<uchar  >(a, cond, b); break;
        case b8:  replace<char   >(a, cond, b); break;
        default:  TYPE_ERROR(2, ainfo.getType());
        }

    } CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
void replace_scalar(af_array a, const af_array cond, const double b)
{
    select_scalar<T, true>(getWritableArray<T>(a), getArray<char>(cond), getArray<T>(a), b);
}

af_err af_replace_scalar(af_array a, const af_array cond, const double b)
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

        switch (ainfo.getType()) {
        case f32: replace_scalar<float  >(a, cond, b); break;
        case f64: replace_scalar<double >(a, cond, b); break;
        case c32: replace_scalar<cfloat >(a, cond, b); break;
        case c64: replace_scalar<cdouble>(a, cond, b); break;
        case s32: replace_scalar<int    >(a, cond, b); break;
        case u32: replace_scalar<uint   >(a, cond, b); break;
        case s64: replace_scalar<intl   >(a, cond, b); break;
        case u64: replace_scalar<uintl  >(a, cond, b); break;
        case u8:  replace_scalar<uchar  >(a, cond, b); break;
        case b8:  replace_scalar<char   >(a, cond, b); break;
        default:  TYPE_ERROR(2, ainfo.getType());
        }

    } CATCHALL;
    return AF_SUCCESS;
}
