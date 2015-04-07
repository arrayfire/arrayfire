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

#include <cusolverDnManager.hpp>
#include <cublas_v2.h>
#include <identity.hpp>
#include <iostream>
#include <memory.hpp>
#include <copy.hpp>

#include <math.hpp>
#include <err_common.hpp>

namespace cuda
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("CUDA Solve not implemented",
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
namespace cuda
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("CUDA cusolver not available. Linear Algebra is disabled",
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
