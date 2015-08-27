/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <wrap.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array wrap(const af_array in,
                            const dim_t ox, const dim_t oy,
                            const dim_t wx, const dim_t wy,
                            const dim_t sx, const dim_t sy,
                            const dim_t px, const dim_t py,
                            const bool is_column)
{
    return getHandle(wrap<T>(getArray<T>(in), ox, oy, wx, wy, sx, sy, px, py, is_column));
}

af_err af_wrap(af_array *out, const af_array in,
               const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy,
               const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py,
               const bool is_column)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        ARG_ASSERT(2, wx > 0);
        ARG_ASSERT(3, wx > 0);
        ARG_ASSERT(4, sx > 0);
        ARG_ASSERT(5, sy > 0);

        dim_t nx = (ox + 2 * px - wx) / sx + 1;
        dim_t ny = (oy + 2 * py - wy) / sy + 1;

        dim_t patch_size  = is_column ? idims[0] : idims[1];
        dim_t num_patches = is_column ? idims[1] : idims[0];

        DIM_ASSERT(1, patch_size == wx * wy);
        DIM_ASSERT(1, num_patches == nx * ny);

        af_array output;

        switch(type) {
            case f32: output = wrap<float  >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case f64: output = wrap<double >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case c32: output = wrap<cfloat >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case c64: output = wrap<cdouble>(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case s32: output = wrap<int    >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u32: output = wrap<uint   >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case s64: output = wrap<intl   >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u64: output = wrap<uintl  >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u8:  output = wrap<uchar  >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case b8:  output = wrap<char   >(in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
