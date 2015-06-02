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
#include <inverse.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array inverse(const af_array in)
{
    return getHandle(inverse<T>(getArray<T>(in)));
}

af_err af_inverse(af_array *out, const af_array in, const af_mat_prop options)
{
    try {
        ArrayInfo i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        if (options != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in inverse", AF_ERR_NOT_SUPPORTED);
        }

        DIM_ASSERT(1, i_info.dims()[0] == i_info.dims()[1]);      // Only square matrices
        ARG_ASSERT(1, i_info.isFloating());                       // Only floating and complex types

        af_array output;

        switch(type) {
            case f32: output = inverse<float  >(in);  break;
            case f64: output = inverse<double >(in);  break;
            case c32: output = inverse<cfloat >(in);  break;
            case c64: output = inverse<cdouble>(in);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
