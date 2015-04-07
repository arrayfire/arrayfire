/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

namespace opencl
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#else

namespace opencl
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#endif

