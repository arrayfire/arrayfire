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
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <resize.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array resize(const af_array in, const dim_t odim0, const dim_t odim1,
                              const af_interp_type method)
{
    return getHandle(resize<T>(getArray<T>(in), odim0, odim1, method));
}

af_err af_resize(af_array *out, const af_array in, const dim_t odim0, const dim_t odim1,
                 const af_interp_type method)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();

        ARG_ASSERT(4, (method == AF_INTERP_BILINEAR ||
                       method == AF_INTERP_NEAREST  ||
                       method == AF_INTERP_LOWER
                       ));
        DIM_ASSERT(2, odim0 > 0);
        DIM_ASSERT(3, odim1 > 0);

        af_array output;

        switch(type) {
            case f32: output = resize<float  >(in, odim0, odim1, method);  break;
            case f64: output = resize<double >(in, odim0, odim1, method);  break;
            case c32: output = resize<cfloat >(in, odim0, odim1, method);  break;
            case c64: output = resize<cdouble>(in, odim0, odim1, method);  break;
            case s32: output = resize<int    >(in, odim0, odim1, method);  break;
            case u32: output = resize<uint   >(in, odim0, odim1, method);  break;
            case s64: output = resize<intl   >(in, odim0, odim1, method);  break;
            case u64: output = resize<uintl  >(in, odim0, odim1, method);  break;
            case u8:  output = resize<uchar  >(in, odim0, odim1, method);  break;
            case b8:  output = resize<char   >(in, odim0, odim1, method);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
