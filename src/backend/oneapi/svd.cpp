/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <blas.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>  // error check functions and Macros
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <svd.hpp>  // oneapi backend function header
#include <transpose.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include "oneapi/mkl/lapack.hpp"

namespace arrayfire {
namespace oneapi {

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    dim4 iDims = in.dims();
    int64_t M  = iDims[0];
    int64_t N  = iDims[1];

    dim4 iStrides = in.strides();
    dim4 uStrides = u.strides();
    dim4 vStrides = vt.strides();
    int64_t LDA   = iStrides[1];
    int64_t LDU   = uStrides[1];
    int64_t LDVt  = vStrides[1];

    int64_t scratch_size =
        ::oneapi::mkl::lapack::gesvd_scratchpad_size<compute_t<T>>(
            getQueue(), ::oneapi::mkl::jobsvd::vectors,
            ::oneapi::mkl::jobsvd::vectors, M, N, LDA, LDU, LDVt);

    auto scratchpad = memAlloc<compute_t<T>>(scratch_size);

    sycl::buffer<compute_t<T>> in_buffer =
        in.template getBufferWithOffset<compute_t<T>>();

    sycl::buffer<compute_t<Tr>> sBuf =
        s.template getBufferWithOffset<compute_t<Tr>>();
    sycl::buffer<compute_t<T>> uBuf =
        u.template getBufferWithOffset<compute_t<T>>();
    sycl::buffer<compute_t<T>> vtBuf =
        vt.template getBufferWithOffset<compute_t<T>>();

    ::oneapi::mkl::lapack::gesvd(getQueue(), ::oneapi::mkl::jobsvd::vectors,
                                 ::oneapi::mkl::jobsvd::vectors, M, N,
                                 in_buffer, LDA, sBuf, uBuf, LDU, vtBuf, LDVt,
                                 *scratchpad, scratchpad->size());
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    Array<T> in_copy = copyArray<T>(in);
    svdInPlace(s, u, vt, in_copy);
}

#define INSTANTIATE(T, Tr)                                               \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace oneapi
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace oneapi {

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T, Tr)                                               \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace oneapi
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
