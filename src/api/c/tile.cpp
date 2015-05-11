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
#include <tile.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array tile(const af_array in, const af::dim4 &tileDims)
{
    return getHandle(tile<T>(getArray<T>(in), tileDims));
}

af_err af_tile(af_array *out, const af_array in, const af::dim4 &tileDims)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();

        DIM_ASSERT(1, info.dims().elements() > 0);
        DIM_ASSERT(2, tileDims.elements() > 0);

        af_array output;

        switch(type) {
            case f32: output = tile<float  >(in, tileDims);  break;
            case c32: output = tile<cfloat >(in, tileDims);  break;
            case f64: output = tile<double >(in, tileDims);  break;
            case c64: output = tile<cdouble>(in, tileDims);  break;
            case b8:  output = tile<char   >(in, tileDims);  break;
            case s32: output = tile<int    >(in, tileDims);  break;
            case u32: output = tile<uint   >(in, tileDims);  break;
            case u8:  output = tile<uchar  >(in, tileDims);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_tile(af_array *out, const af_array in,
               const unsigned x, const unsigned y,
               const unsigned z, const unsigned w)
{
    af::dim4 tileDims(x, y, z, w);
    return af_tile(out, in, tileDims);
}
