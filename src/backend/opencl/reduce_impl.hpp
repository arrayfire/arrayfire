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
    Array<To>* reduce(const Array<Ti> &in, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;
        Array<To> *out = createEmptyArray<To>(odims);
        kernel::reduce<Ti, To, op>(*out, in, dim);
        return out;
    }

    template<af_op_t op, typename Ti, typename To>
    To reduce_global(const Array<Ti> &in)
    {
        return kernel::reduce_global<Ti, To, op>(in);
    }
}

#define INSTANTIATE(Op, Ti, To)                                         \
    template Array<To>* reduce<Op, Ti, To>(const Array<Ti> &in, const int dim); \
    template To reduce_global<Op, Ti, To>(const Array<Ti> &in);
