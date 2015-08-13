/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <iostream>
#include <af/data.h>
#include <af/arith.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <tile.hpp>
#include <arith.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array tile(const af_array in, const af::dim4 &tileDims)
{
    const Array<T> inArray = getArray<T>(in);
    const dim4 inDims = inArray.dims();


    // FIXME: Always use JIT instead of checking for the condition.
    // The current limitation exists for performance reasons. it should change in the future.

    bool take_jit_path = true;
    dim4 outDims(1, 1, 1, 1);

    // Check if JIT path can be taken. JIT path can only be taken if tiling a singleton dimension.
    for (int i = 0; i < 4; i++) {
        take_jit_path &= (inDims[i] == 1 || tileDims[i] == 1);
        outDims[i] = inDims[i] * tileDims[i];
    }

    if (take_jit_path) {
        // FIXME: This Should ideally call a NOP function, but adding 0 should be OK
        // This does not allocate any memory, just a JIT node
        Array<T> tmpArray = createValueArray<T>(outDims, scalar<T>(0));
        return getHandle(arithOp<T, af_add_t>(inArray, tmpArray, outDims));
    } else {
        return getHandle(tile<T>(inArray, tileDims));
    }
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
            case s64: output = tile<intl   >(in, tileDims);  break;
            case u64: output = tile<uintl  >(in, tileDims);  break;
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
