/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>
#include <err_common.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <af/dim4.hpp>
#include <handle.hpp>
#include <iostream>
#include <cassert>
#include <err_cpu.hpp>

#include <lapack_helper.hpp>

namespace cpu
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("CPU Solve not implemented",
              AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#else

namespace cpu
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("Linear Algebra is diabled on CPU",
              AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#endif

