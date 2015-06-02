/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <regions.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_array regions(af_array const &in, af_connectivity connectivity)
{
    return getHandle<T>(regions<T>(getArray<char>(in), connectivity));
}

af_err af_regions(af_array *out, const af_array in, const af_connectivity connectivity, const af_dtype type)
{
    try {
        ARG_ASSERT(2, (connectivity==AF_CONNECTIVITY_4 || connectivity==AF_CONNECTIVITY_8));

        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        af_dtype in_type = info.getType();
        if (in_type != b8) {
            TYPE_ERROR(1, in_type);
        }

        af_array output;
        switch(type) {
            case f32: output = regions<float >(in, connectivity); break;
            case f64: output = regions<double>(in, connectivity); break;
            case s32: output = regions<int   >(in, connectivity); break;
            case u32: output = regions<uint  >(in, connectivity); break;
            default : TYPE_ERROR(0, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
