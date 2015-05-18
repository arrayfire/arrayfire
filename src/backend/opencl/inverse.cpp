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
#include <identity.hpp>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)

namespace opencl
{

template<typename T>
Array<T> inverse(const Array<T> &in)
{
    Array<T> I = identity<T>(in.dims());
    return solve<T>(in, I);
}

#define INSTANTIATE(T)                                                                   \
    template Array<T> inverse<T> (const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}

#else

namespace opencl
{

template<typename T>
Array<T> inverse(const Array<T> &in)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T)                                                                   \
    template Array<T> inverse<T> (const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}

#endif
