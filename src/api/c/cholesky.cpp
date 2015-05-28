/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <cholesky.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array cholesky(int *info, const af_array in, const bool is_upper)
{
    return getHandle(cholesky<T>(info, getArray<T>(in), is_upper));
}

template<typename T>
static inline int cholesky_inplace(af_array in, const bool is_upper)
{
     return cholesky_inplace<T>(getWritableArray<T>(in), is_upper);
}

af_err af_cholesky(af_array *out, int *info, const af_array in, const bool is_upper)
{
    try {
        ArrayInfo i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("cholesky can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        ARG_ASSERT(2, i_info.isFloating());                  // Only floating and complex types
        DIM_ASSERT(1, i_info.dims()[0] == i_info.dims()[1]); // Only square matrices

        af_array output;
        switch(type) {
            case f32: output = cholesky<float  >(info, in, is_upper);  break;
            case f64: output = cholesky<double >(info, in, is_upper);  break;
            case c32: output = cholesky<cfloat >(info, in, is_upper);  break;
            case c64: output = cholesky<cdouble>(info, in, is_upper);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_cholesky_inplace(int *info, af_array in, const bool is_upper)
{
    try {
        ArrayInfo i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("cholesky can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating()); // Only floating and complex types
        DIM_ASSERT(1, i_info.dims()[0] == i_info.dims()[1]); // Only square matrices

        int out;

        switch(type) {
            case f32: out = cholesky_inplace<float  >(in, is_upper);  break;
            case f64: out = cholesky_inplace<double >(in, is_upper);  break;
            case c32: out = cholesky_inplace<cfloat >(in, is_upper);  break;
            case c64: out = cholesky_inplace<cdouble>(in, is_upper);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*info, out);
    }
    CATCHALL;

    return AF_SUCCESS;
}
