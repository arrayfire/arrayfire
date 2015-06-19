/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <unwrap.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array unwrap(const af_array in, const dim_t wx, const dim_t wy,
                              const dim_t sx, const dim_t sy, const dim_t px, const dim_t py)
{
    return getHandle(unwrap<T>(getArray<T>(in), wx, wy, sx, sy, px, py));
}

af_err af_unwrap(af_array *out, const af_array in, const dim_t wx, const dim_t wy,
                 const dim_t sx, const dim_t sy, const dim_t px, const dim_t py)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        DIM_ASSERT(2, wx > 0 && wx <= idims[0] + px);
        DIM_ASSERT(3, wy > 0 && wy <= idims[1] + py);
        ARG_ASSERT(4, sx > 0);
        ARG_ASSERT(5, sy > 0);
        ARG_ASSERT(6, px >= 0 && px < wx);
        ARG_ASSERT(7, py >= 0 && py < wy);

        af_array output;

        switch(type) {
            case f32: output = unwrap<float  >(in, wx, wy, sx, sy, px, py);  break;
            case f64: output = unwrap<double >(in, wx, wy, sx, sy, px, py);  break;
            case c32: output = unwrap<cfloat >(in, wx, wy, sx, sy, px, py);  break;
            case c64: output = unwrap<cdouble>(in, wx, wy, sx, sy, px, py);  break;
            case s32: output = unwrap<int    >(in, wx, wy, sx, sy, px, py);  break;
            case u32: output = unwrap<uint   >(in, wx, wy, sx, sy, px, py);  break;
            case s64: output = unwrap<intl   >(in, wx, wy, sx, sy, px, py);  break;
            case u64: output = unwrap<uintl  >(in, wx, wy, sx, sy, px, py);  break;
            case u8:  output = unwrap<uchar  >(in, wx, wy, sx, sy, px, py);  break;
            case b8:  output = unwrap<char   >(in, wx, wy, sx, sy, px, py);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
