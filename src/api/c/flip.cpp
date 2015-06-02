/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <vector>
#include <cassert>

#include <af/array.h>
#include <af/data.h>
#include <af/index.h>
#include <af/seq.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <lookup.hpp>

using namespace detail;
using std::vector;
using std::swap;

template<typename T>
static af_array flipArray(const af_array in, const unsigned dim)
{
    const Array<T> &input = getArray<T>(in);
    vector<af_seq> index(4);

    for (int i = 0; i < 4; i++) {
        index[i] = af_span;
    }

    // Reverse "dim"
    dim4 in_dims = input.dims();
    af_seq s = {(double)(in_dims[dim] - 1), 0, -1};

    index[dim] = s;

    Array<T> dst =  createSubArray(input, index);

    return getHandle(dst);
}

af_err af_flip(af_array *result, const af_array in, const unsigned dim)
{
    af_array out;
    try {
        ArrayInfo in_info = getInfo(in);

        if (in_info.ndims() <= dim) {
            *result = retain(in);
            return AF_SUCCESS;
        }

        af_dtype in_type = in_info.getType();

        switch(in_type) {
        case f32:    out = flipArray<float>   (in, dim);  break;
        case c32:    out = flipArray<cfloat>  (in, dim);  break;
        case f64:    out = flipArray<double>  (in, dim);  break;
        case c64:    out = flipArray<cdouble> (in, dim);  break;
        case b8:     out = flipArray<char>    (in, dim);  break;
        case s32:    out = flipArray<int>     (in, dim);  break;
        case u32:    out = flipArray<unsigned>(in, dim);  break;
        case u8:     out = flipArray<uchar>   (in, dim);  break;
        default:    TYPE_ERROR(1, in_type);
        }
    }
    CATCHALL

    swap(*result, out);
    return AF_SUCCESS;
}
