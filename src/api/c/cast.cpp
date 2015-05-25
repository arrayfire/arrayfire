/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>

#include <cast.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>

using namespace detail;

static af_array cast(const af_array in, const af_dtype type)
{
    const ArrayInfo info = getInfo(in);

    if (info.getType() == type) {
        return retain(in);
    }

    switch (type) {
    case f32: return getHandle(castArray<float   >(in));
    case f64: return getHandle(castArray<double  >(in));
    case c32: return getHandle(castArray<cfloat  >(in));
    case c64: return getHandle(castArray<cdouble >(in));
    case s32: return getHandle(castArray<int     >(in));
    case u32: return getHandle(castArray<uint    >(in));
    case u8 : return getHandle(castArray<uchar   >(in));
    case b8 : return getHandle(castArray<char    >(in));
    case s64: return getHandle(castArray<intl    >(in));
    case u64: return getHandle(castArray<uintl   >(in));
    default: TYPE_ERROR(2, type);
    }
}

af_err af_cast(af_array *out, const af_array in, const af_dtype type)
{
    try {
        af_array res = cast(in, type);
        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_cplx(af_array *out, const af_array in, const af_dtype type)
{
    try {
        af_array res;
        ArrayInfo in_info = getInfo(in);

        if (in_info.isDouble()) {
            res = cast(in, c64);
        } else {
            res = cast(in, c32);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
