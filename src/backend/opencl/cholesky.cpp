/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cholesky.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_cholesky.hpp>
#include <magma/magma.h>
#include <triangle.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    if (OpenCLCPUOffload()) { return cpu::cholesky_inplace(in, is_upper); }

    dim4 iDims = in.dims();
    int N      = iDims[0];

    magma_uplo_t uplo = is_upper ? MagmaUpper : MagmaLower;

    int info           = 0;
    cl::Buffer *in_buf = in.get();
    magma_potrf_gpu<T>(uplo, N, (*in_buf)(), in.getOffset(), in.strides()[1],
                       getQueue()(), &info);
    return info;
}

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    if (OpenCLCPUOffload()) { return cpu::cholesky(info, in, is_upper); }

    Array<T> out = copyArray<T>(in);
    *info        = cholesky_inplace(out, is_upper);

    triangle<T>(out, out, is_upper, false);

    return out;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace opencl
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
