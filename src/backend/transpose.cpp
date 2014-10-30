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
#include <af/index.h>
#include <af/blas.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <transpose.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array transpose(const af_array in)
{
    return getHandle(*transpose<T>(getArray<T>(in)));
}

af_err af_transpose(af_array *out, af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 dims = info.dims();

        DIM_ASSERT(1, (dims.ndims()<=3));

        if (dims[0]==1 || dims[1]==1) {
            // for a vector OR a batch of vectors
            // we can use modDims to transpose
            af::dim4 outDims(dims[1],dims[0],dims[2],dims[3]);
            return af_moddims(out, in, outDims.ndims(), outDims.get());
        }

        af_array output;
        switch(type) {
            case f32: output = transpose<float>(in);          break;
            case c32: output = transpose<cfloat>(in);         break;
            case f64: output = transpose<double>(in);         break;
            case c64: output = transpose<cdouble>(in);        break;
            case b8 : output = transpose<char>(in);           break;
            case s32: output = transpose<int>(in);            break;
            case u32: output = transpose<uint>(in);           break;
            case u8 : output = transpose<uchar>(in);          break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
