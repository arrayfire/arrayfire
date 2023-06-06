/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cholesky.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <platform.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <memory.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <triangle.hpp>
#include <algorithm>

namespace arrayfire {
namespace oneapi {

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    dim4 iDims    = in.dims();
    dim4 iStrides = in.strides();
    int64_t N     = iDims[0];
    int64_t LDA   = iStrides[1];

    int64_t lwork = 0;

    ::oneapi::mkl::uplo uplo = ::oneapi::mkl::uplo::lower;
    if (is_upper) { uplo = ::oneapi::mkl::uplo::upper; }

    lwork = ::oneapi::mkl::lapack::potrf_scratchpad_size<compute_t<T>>(
        getQueue(), uplo, N, LDA);

    auto workspace = memAlloc<compute_t<T>>(std::max<int64_t>(lwork, 1));
    sycl::buffer<compute_t<T>> in_buffer =
        in.template getBufferWithOffset<compute_t<T>>();

    try {
        ::oneapi::mkl::lapack::potrf(getQueue(), uplo, N, in_buffer, LDA,
                                     *workspace, workspace->size());
    } catch (::oneapi::mkl::lapack::exception const &e) {
        AF_ERROR(
            "Unexpected exception caught during synchronous\
                call to LAPACK API",
            AF_ERR_RUNTIME);
        return e.info();
    }

    return 0;
}

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
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

}  // namespace oneapi
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OneAPI backend",
             AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OneAPI backend",
             AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
