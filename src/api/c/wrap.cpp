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
#include <handle.hpp>
#include <wrap.hpp>
#include <af/defines.h>
#include <af/image.h>

using af::dim4;
using namespace detail;

template<typename T>
static inline void wrap(af_array *out, const af_array in,
                        const dim_t ox, const dim_t oy,
                        const dim_t wx, const dim_t wy,
                        const dim_t sx, const dim_t sy,
                        const dim_t px, const dim_t py,
                        const bool is_column)
{
    wrap<T>(getWritableArray<T>(*out), getArray<T>(in), ox, oy, wx, wy, sx, sy, px, py, is_column);
}

af_err af_wrap(af_array* out, const af_array in, const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py, const bool is_column) {
    try {
        const ArrayInfo& info = getInfo(in);
        const af_dtype in_type = info.getType();
        const dim4 in_dims = info.dims();

        ARG_ASSERT(4, wx > 0);
        ARG_ASSERT(5, wy > 0);
        ARG_ASSERT(6, sx > 0);
        ARG_ASSERT(7, sy > 0);

        const dim_t nx = (ox + 2 * px - wx) / sx + 1;
        const dim_t ny = (oy + 2 * py - wy) / sy + 1;

        const dim_t patch_size  = is_column ? in_dims[0] : in_dims[1];
        const dim_t num_patches = is_column ? in_dims[1] : in_dims[0];

        DIM_ASSERT(1, patch_size == wx * wy);
        DIM_ASSERT(1, num_patches == nx * ny);

        if (*out == 0) {
            *out = createHandle(dim4(ox, oy, in_dims[2], in_dims[3]), in_type);
        } else {
            // \TODO
        }

        switch(in_type) {
            case f32: wrap<float  >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case f64: wrap<double >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case c32: wrap<cfloat >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case c64: wrap<cdouble>(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case s32: wrap<int    >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u32: wrap<uint   >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case s64: wrap<intl   >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u64: wrap<uintl  >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case s16: wrap<short  >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u16: wrap<ushort >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case u8:  wrap<uchar  >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            case b8:  wrap<char   >(out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);  break;
            default:  TYPE_ERROR(1, in_type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
