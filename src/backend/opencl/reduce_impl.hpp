/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <reduce.hpp>
#include <kernel/reduce.hpp>
#include <err_opencl.hpp>

using std::swap;
using af::dim4;
namespace opencl
{
    template<af_op_t op, typename Ti, typename To>
    Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan, double nanval)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;
        Array<To> out = createEmptyArray<To>(odims);
        kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
        return out;
    }

    template<af_op_t op, typename Ti, typename To>
    To reduce_all(const Array<Ti> &in, bool change_nan, double nanval)
    {
        return kernel::reduce_all<Ti, To, op>(in, change_nan, nanval);
    }
}

#define INSTANTIATE(Op, Ti, To)                                         \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim, \
                                          bool change_nan, double nanval); \
    template To reduce_all<Op, Ti, To>(const Array<Ti> &in, bool change_nan, double nanval);
