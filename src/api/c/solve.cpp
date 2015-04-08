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
#include <solve.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array solve(const af_array a, const af_array b, const af_solve_t options)
{
    return getHandle(solve<T>(getArray<T>(a), getArray<T>(b), options));
}

af_err af_solve(af_array *out, const af_array a, const af_array b, const af_solve_t options)
{
    try {
        ArrayInfo a_info = getInfo(a);
        ArrayInfo b_info = getInfo(b);

        af_dtype type = a_info.getType();

        dim4 adims = a_info.dims();
        dim4 bdims = a_info.dims();

        ARG_ASSERT(1, a_info.isFloating());                       // Only floating and complex types
        ARG_ASSERT(2, b_info.isFloating());                       // Only floating and complex types

        DIM_ASSERT(1, bdims[0] == adims[0]);
        DIM_ASSERT(1, bdims[2] == adims[2]);
        DIM_ASSERT(1, bdims[3] == adims[3]);

        af_array output;

        switch(type) {
            case f32: output = solve<float  >(a, b, options);  break;
            case f64: output = solve<double >(a, b, options);  break;
            case c32: output = solve<cfloat >(a, b, options);  break;
            case c64: output = solve<cdouble>(a, b, options);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
