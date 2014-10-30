/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/data.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ops.hpp>
#include <where.hpp>
#include <backend.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array where(const af_array in)
{
    return getHandle(*where<T>(getArray<T>(in)));
}

af_err af_where(af_array *idx, const af_array in)
{
    try {
        af_dtype type = getInfo(in).getType();
        af_array res;
        switch(type) {
        case f32: res = where<float  >(in); break;
        case f64: res = where<double >(in); break;
        case c32: res = where<cfloat >(in); break;
        case c64: res = where<cdouble>(in); break;
        case s32: res = where<int    >(in); break;
        case u32: res = where<uint   >(in); break;
        case u8 : res = where<uchar  >(in); break;
        case s8 : res = where<char   >(in); break;
        case b8 : res = where<uchar  >(in); break;
        default:
            TYPE_ERROR(1, type);
        }
        std::swap(*idx, res);
    }
    CATCHALL

    return AF_SUCCESS;
}
