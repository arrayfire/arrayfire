/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <err_oneapi.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <blas.hpp>
#include <copy.hpp>
#include <identity.hpp>
#include <kernel/triangle.hpp>
#include <memory.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace oneapi {

using sycl::buffer;

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    Array<T> in_copy = copyArray<T>(in);

    // Get workspace needed for QR
    std::int64_t scratchpad_size =
        ::oneapi::mkl::lapack::geqrf_scratchpad_size<compute_t<T>>(
            getQueue(), iDims[0], iDims[1], in_copy.strides()[1]);

    auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);

    t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    buffer<compute_t<T>> iBuf =
        in_copy.template getBufferWithOffset<compute_t<T>>();
    buffer<compute_t<T>> tBuf = t.template getBufferWithOffset<compute_t<T>>();
    ::oneapi::mkl::lapack::geqrf(getQueue(), M, N, iBuf, in_copy.strides()[1],
                                 tBuf, *scratchpad, scratchpad->size());
    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    constexpr bool is_upper     = true;
    constexpr bool is_unit_diag = false;
    kernel::triangle<T>(r, in_copy, is_upper, is_unit_diag);

    int mn = max(M, N);
    dim4 qdims(M, mn);
    q = identity<T>(qdims);

    buffer<compute_t<T>> qBuf = q.template getBufferWithOffset<compute_t<T>>();
    if constexpr (std::is_floating_point<compute_t<T>>()) {
        std::int64_t scratchpad_size =
            ::oneapi::mkl::lapack::ormqr_scratchpad_size<compute_t<T>>(
                getQueue(), ::oneapi::mkl::side::left,
                ::oneapi::mkl::transpose::nontrans, q.dims()[0], q.dims()[1],
                min(M, N), in_copy.strides()[1], q.strides()[1]);

        auto scratchpad_ormqr = memAlloc<compute_t<T>>(scratchpad_size);
        ::oneapi::mkl::lapack::ormqr(
            getQueue(), ::oneapi::mkl::side::left,
            ::oneapi::mkl::transpose::nontrans, q.dims()[0], q.dims()[1],
            min(M, N), iBuf, in_copy.strides()[1], tBuf, qBuf, q.strides()[1],
            *scratchpad_ormqr, scratchpad_ormqr->size());

    } else if constexpr (common::isComplex(static_cast<af::dtype>(
                             dtype_traits<compute_t<T>>::af_type))) {
        std::int64_t scratchpad_size =
            ::oneapi::mkl::lapack::unmqr_scratchpad_size<compute_t<T>>(
                getQueue(), ::oneapi::mkl::side::left,
                ::oneapi::mkl::transpose::nontrans, q.dims()[0], q.dims()[1],
                min(M, N), in_copy.strides()[1], q.strides()[1]);

        auto scratchpad_ormqr = memAlloc<compute_t<T>>(scratchpad_size);
        ::oneapi::mkl::lapack::unmqr(
            getQueue(), ::oneapi::mkl::side::left,
            ::oneapi::mkl::transpose::nontrans, q.dims()[0], q.dims()[1],
            min(M, N), iBuf, in_copy.strides()[1], tBuf, qBuf, q.strides()[1],
            *scratchpad_ormqr, scratchpad_ormqr->size());
    }
    q.resetDims(dim4(M, M));
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims    = in.dims();
    dim4 iStrides = in.strides();
    int M         = iDims[0];
    int N         = iDims[1];

    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    // Get workspace needed for QR
    std::int64_t scratchpad_size =
        ::oneapi::mkl::lapack::geqrf_scratchpad_size<compute_t<T>>(
            getQueue(), iDims[0], iDims[1], iStrides[1]);

    auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);

    buffer<compute_t<T>> iBuf = in.template getBufferWithOffset<compute_t<T>>();
    buffer<compute_t<T>> tBuf = t.template getBufferWithOffset<compute_t<T>>();
    // In place Perform in place QR
    ::oneapi::mkl::lapack::geqrf(getQueue(), iDims[0], iDims[1], iBuf,
                                 iStrides[1], tBuf, *scratchpad,
                                 scratchpad->size());
    return t;
}

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace oneapi {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
