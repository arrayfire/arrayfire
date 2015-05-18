/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/data.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <shift.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array shift(const af_array in, const int sdims[4])
{
    return getHandle(shift<T>(getArray<T>(in), sdims));
}

af_err af_shift(af_array *out, const af_array in, const int sdims[4])
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();

        DIM_ASSERT(1, info.elements() > 0);

        af_array output;

        switch(type) {
            case f32: output = shift<float  >(in, sdims);  break;
            case c32: output = shift<cfloat >(in, sdims);  break;
            case f64: output = shift<double >(in, sdims);  break;
            case c64: output = shift<cdouble>(in, sdims);  break;
            case b8:  output = shift<char   >(in, sdims);  break;
            case s32: output = shift<int    >(in, sdims);  break;
            case u32: output = shift<uint   >(in, sdims);  break;
            case u8:  output = shift<uchar  >(in, sdims);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_shift(af_array *out, const af_array in,
                const int x, const int y, const int z, const int w)
{
    const int sdims[] = {x, y, z, w};
    return af_shift(out, in, sdims);
}
