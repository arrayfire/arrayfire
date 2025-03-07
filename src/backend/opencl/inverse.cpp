/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_opencl.hpp>
#include <identity.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_inverse.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    if (OpenCLCPUOffload()) {
        if (in.dims()[0] == in.dims()[1]) { return cpu::inverse(in); }
    }
    Array<T> I = identity<T>(in.dims());
    return solve<T>(in, I);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#endif
