/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/algorithm.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ops.hpp>
#include <scan.hpp>
#include <backend.hpp>

using af::dim4;
using namespace detail;

template<af_op_t op, typename Ti, typename To>
static inline af_array scan(const af_array in, const int dim)
{
    return getHandle(scan<op,Ti,To>(getArray<Ti>(in), dim));
}


af_err af_accum(af_array *out, const af_array in, const int dim)
{
    ARG_ASSERT(2, dim >= 0);
    ARG_ASSERT(2, dim <  4);

    try {

        const ArrayInfo& in_info = getInfo(in);

        if (dim >= (int)in_info.ndims()) {
            *out = retain(in);
            return AF_SUCCESS;
        }

        af_dtype type = in_info.getType();
        af_array res;

        switch(type) {
        case f32:  res = scan<af_add_t, float  , float  >(in, dim); break;
        case f64:  res = scan<af_add_t, double , double >(in, dim); break;
        case c32:  res = scan<af_add_t, cfloat , cfloat >(in, dim); break;
        case c64:  res = scan<af_add_t, cdouble, cdouble>(in, dim); break;
        case u32:  res = scan<af_add_t, uint   , uint   >(in, dim); break;
        case s32:  res = scan<af_add_t, int    , int    >(in, dim); break;
        case u64:  res = scan<af_add_t, uintl  , uintl  >(in, dim); break;
        case s64:  res = scan<af_add_t, intl   , intl   >(in, dim); break;
        case u8:   res = scan<af_add_t, uchar  , uint   >(in, dim); break;
        // Make sure you are adding only "1" for every non zero value, even if op == af_add_t
        case b8:   res = scan<af_notzero_t, char  , uint   >(in, dim); break;
        default:
            TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
