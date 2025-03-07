/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <err_oneapi.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <Array.hpp>
#include <blas.hpp>
#include <common/cast.hpp>
#include <copy.hpp>
#include <lu.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <platform.hpp>
#include <transpose.hpp>

#include <common/traits.hpp>
#include <algorithm>
#include <type_traits>
#include <vector>

using arrayfire::common::cast;
using std::min;
using std::vector;
using sycl::buffer;

namespace arrayfire {
namespace oneapi {

static ::oneapi::mkl::transpose toMKLTranspose(af_mat_prop opt) {
    switch (opt) {
        case AF_MAT_NONE: return ::oneapi::mkl::transpose::nontrans;
        case AF_MAT_TRANS: return ::oneapi::mkl::transpose::trans;
        case AF_MAT_CTRANS: return ::oneapi::mkl::transpose::conjtrans;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
}

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    const int64_t N    = A.dims()[0];
    const int64_t NRHS = b.dims()[1];
    const int64_t LDA  = A.strides()[1];
    const int64_t LDB  = b.strides()[1];

    ::oneapi::mkl::transpose opts = toMKLTranspose(options);
    std::int64_t scratchpad_size =
        ::oneapi::mkl::lapack::getrs_scratchpad_size<compute_t<T>>(
            getQueue(), opts, N, NRHS, LDA, LDB);

    Array<intl> ipiv        = cast<intl, int>(pivot);
    buffer<int64_t> ipivBuf = ipiv.get()->reinterpret<int64_t, 1>();
    auto scratchpad         = memAlloc<compute_t<T>>(scratchpad_size);

    Array<compute_t<T>> B     = copyArray<compute_t<T>>(b);
    buffer<compute_t<T>> aBuf = A.template getBufferWithOffset<compute_t<T>>();
    buffer<compute_t<T>> bBuf = B.template getBufferWithOffset<compute_t<T>>();

    ::oneapi::mkl::lapack::getrs(getQueue(), opts, N, NRHS, aBuf, LDA, ipivBuf,
                                 bBuf, LDB, *scratchpad, scratchpad->size());
    return B;
}

template<typename T>
Array<T> generalSolve(const Array<T> &a, const Array<T> &b) {
    int batches = a.dims()[2] * a.dims()[3];

    dim4 aDims = a.dims();
    dim4 bDims = b.dims();
    int M      = aDims[0];
    int N      = aDims[1];
    int K      = bDims[1];
    int MN     = std::min(M, N);

    int lda        = a.strides()[1];
    int astride    = a.strides()[2];
    auto ipiv      = memAlloc<int64_t>(MN * batches);
    int ipivstride = MN;

    int ldb     = b.strides()[1];
    int bstride = b.strides()[2];

    vector<int> info(batches, 0);

    Array<T> A = copyArray<T>(a);  // A will be overwritten by L,U
    Array<T> B = copyArray<T>(b);  // will be overwritten with solution

    std::int64_t scratchpad_size =
        ::oneapi::mkl::lapack::getrf_batch_scratchpad_size<compute_t<T>>(
            getQueue(), M, N, lda, astride, ipivstride, batches);

    auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);

    buffer<compute_t<T>> aBuf = A.template getBufferWithOffset<compute_t<T>>();
    buffer<compute_t<T>> bBuf = B.template getBufferWithOffset<compute_t<T>>();
    ::oneapi::mkl::lapack::getrf_batch(getQueue(), M, N, aBuf, lda, astride,
                                       *ipiv, ipivstride, batches, *scratchpad,
                                       scratchpad->size());

    scratchpad_size =
        ::oneapi::mkl::lapack::getrs_batch_scratchpad_size<compute_t<T>>(
            getQueue(), ::oneapi::mkl::transpose::nontrans, N, K, lda, astride,
            ipivstride, ldb, bstride, batches);

    auto scratchpad_rs = memAlloc<compute_t<T>>(scratchpad_size);

    ::oneapi::mkl::lapack::getrs_batch(
        getQueue(), ::oneapi::mkl::transpose::nontrans, N, K, aBuf, lda,
        astride, *ipiv, ipivstride, bBuf, ldb, bstride, batches, *scratchpad_rs,
        scratchpad_rs->size());

    return B;
}

template<typename T>
Array<T> leastSquares(const Array<T> &a, const Array<T> &b) {
    int64_t M  = a.dims()[0];
    int64_t N  = a.dims()[1];
    int64_t K  = b.dims()[1];
    int64_t MN = min(M, N);

    Array<T> B = createEmptyArray<T>(dim4());

    if (M < N) {
        const dim4 NullShape(0, 0, 0, 0);

        // Least squres for this case is solved using the following
        // solve(A, B) == matmul(Q, Xpad);
        // Where:
        // Xpad == pad(Xt, N - M, 1);
        // Xt   == tri_solve(R1, B);
        // R1   == R(seq(M), seq(M));
        // transpose(A) == matmul(Q, R);

        // QR is performed on the transpose of A
        Array<T> A = transpose<T>(a, true);
        dim4 endPadding(N - b.dims()[0], K - b.dims()[1], 0, 0);
        B = (endPadding == NullShape
                 ? copyArray(b)
                 : padArrayBorders(b, NullShape, endPadding, AF_PAD_ZERO));

        // Get workspace needed for QR
        std::int64_t scratchpad_size =
            ::oneapi::mkl::lapack::geqrf_scratchpad_size<compute_t<T>>(
                getQueue(), A.dims()[0], A.dims()[1], A.strides()[1]);

        auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);
        auto t          = memAlloc<compute_t<T>>(MN);

        buffer<compute_t<T>> aBuf =
            A.template getBufferWithOffset<compute_t<T>>();
        // In place Perform in place QR
        ::oneapi::mkl::lapack::geqrf(getQueue(), A.dims()[0], A.dims()[1], aBuf,
                                     A.strides()[1], *t, *scratchpad,
                                     scratchpad->size());

        // R1 = R(seq(M), seq(M));
        A.resetDims(dim4(M, M));

        // Bt = tri_solve(R1, B);
        B.resetDims(dim4(M, K));

        buffer<compute_t<T>> bBuf =
            B.template getBufferWithOffset<compute_t<T>>();
        // TODO: move to helper? trsm<T>(A, B, AF_MAT_CTRANS, true, true,
        // false);
        compute_t<T> alpha = scalar<compute_t<T>>(1);
        ::oneapi::mkl::blas::trsm(
            getQueue(), ::oneapi::mkl::side::left, ::oneapi::mkl::uplo::upper,
            ::oneapi::mkl::transpose::conjtrans, ::oneapi::mkl::diag::nonunit,
            B.dims()[0], B.dims()[1], alpha, aBuf, A.strides()[1], bBuf,
            B.strides()[1]);

        // Bpad = pad(Bt, ..)
        B.resetDims(dim4(N, K));

        // matmul(Q, Bpad)
        if constexpr (std::is_floating_point<compute_t<T>>()) {
            std::int64_t scratchpad_size =
                ::oneapi::mkl::lapack::ormqr_scratchpad_size<compute_t<T>>(
                    getQueue(), ::oneapi::mkl::side::left,
                    ::oneapi::mkl::transpose::nontrans, B.dims()[0],
                    B.dims()[1], A.dims()[0], A.strides()[1], B.strides()[1]);

            auto scratchpad_ormqr = memAlloc<compute_t<T>>(scratchpad_size);
            ::oneapi::mkl::lapack::ormqr(
                getQueue(), ::oneapi::mkl::side::left,
                ::oneapi::mkl::transpose::nontrans, B.dims()[0], B.dims()[1],
                A.dims()[0], aBuf, A.strides()[1], *t, bBuf, B.strides()[1],
                *scratchpad_ormqr, scratchpad_ormqr->size());
        } else if constexpr (common::isComplex(static_cast<af::dtype>(
                                 dtype_traits<compute_t<T>>::af_type))) {
            std::int64_t scratchpad_size =
                ::oneapi::mkl::lapack::unmqr_scratchpad_size<compute_t<T>>(
                    getQueue(), ::oneapi::mkl::side::left,
                    ::oneapi::mkl::transpose::nontrans, B.dims()[0],
                    B.dims()[1], A.dims()[0], A.strides()[1], B.strides()[1]);

            auto scratchpad_unmqr = memAlloc<compute_t<T>>(scratchpad_size);
            ::oneapi::mkl::lapack::unmqr(
                getQueue(), ::oneapi::mkl::side::left,
                ::oneapi::mkl::transpose::nontrans, B.dims()[0], B.dims()[1],
                A.dims()[0], aBuf, A.strides()[1], *t, bBuf, B.strides()[1],
                *scratchpad_unmqr, scratchpad_unmqr->size());
        }

    } else if (M > N) {
        // Least squres for this case is solved using the following
        // solve(A, B) == tri_solve(R1, Bt);
        // Where:
        // R1 == R(seq(N), seq(N));
        // Bt == matmul(transpose(Q1), B);
        // Q1 == Q(span, seq(N));
        // A  == matmul(Q, R);

        Array<T> A = copyArray<T>(a);
        B          = copyArray(b);

        // Get workspace needed for QR
        std::int64_t scratchpad_size =
            ::oneapi::mkl::lapack::geqrf_scratchpad_size<compute_t<T>>(
                getQueue(), M, N, A.strides()[1]);

        auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);
        auto t          = memAlloc<compute_t<T>>(MN);

        buffer<compute_t<T>> aBuf =
            A.template getBufferWithOffset<compute_t<T>>();
        // In place Perform in place QR
        ::oneapi::mkl::lapack::geqrf(getQueue(), M, N, aBuf, A.strides()[1], *t,
                                     *scratchpad, scratchpad->size());

        // matmul(Q1, B)
        buffer<compute_t<T>> bBuf =
            B.template getBufferWithOffset<compute_t<T>>();
        if constexpr (std::is_floating_point<compute_t<T>>()) {
            std::int64_t scratchpad_size =
                ::oneapi::mkl::lapack::ormqr_scratchpad_size<compute_t<T>>(
                    getQueue(), ::oneapi::mkl::side::left,
                    ::oneapi::mkl::transpose::trans, M, K, N, A.strides()[1],
                    b.strides()[1]);

            auto scratchpad_ormqr = memAlloc<compute_t<T>>(scratchpad_size);
            ::oneapi::mkl::lapack::ormqr(getQueue(), ::oneapi::mkl::side::left,
                                         ::oneapi::mkl::transpose::trans, M, K,
                                         N, aBuf, A.strides()[1], *t, bBuf,
                                         b.strides()[1], *scratchpad_ormqr,
                                         scratchpad_ormqr->size());
        } else if constexpr (common::isComplex(static_cast<af::dtype>(
                                 dtype_traits<compute_t<T>>::af_type))) {
            std::int64_t scratchpad_size =
                ::oneapi::mkl::lapack::unmqr_scratchpad_size<compute_t<T>>(
                    getQueue(), ::oneapi::mkl::side::left,
                    ::oneapi::mkl::transpose::conjtrans, M, K, N,
                    A.strides()[1], b.strides()[1]);

            auto scratchpad_unmqr = memAlloc<compute_t<T>>(scratchpad_size);
            ::oneapi::mkl::lapack::unmqr(getQueue(), ::oneapi::mkl::side::left,
                                         ::oneapi::mkl::transpose::conjtrans, M,
                                         K, N, aBuf, A.strides()[1], *t, bBuf,
                                         b.strides()[1], *scratchpad_unmqr,
                                         scratchpad_unmqr->size());
        }

        // tri_solve(R1, Bt)
        A.resetDims(dim4(N, N));
        B.resetDims(dim4(N, K));

        compute_t<T> alpha = scalar<compute_t<T>>(1);
        ::oneapi::mkl::blas::trsm(
            getQueue(), ::oneapi::mkl::side::left, ::oneapi::mkl::uplo::upper,
            ::oneapi::mkl::transpose::nontrans, ::oneapi::mkl::diag::nonunit, N,
            K, alpha, aBuf, A.strides()[1], bBuf, B.strides()[1]);
    }

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    Array<compute_t<T>> B = copyArray<T>(b);

    compute_t<T> alpha       = scalar<compute_t<T>>(1);
    ::oneapi::mkl::uplo uplo = (options & AF_MAT_UPPER)
                                   ? ::oneapi::mkl::uplo::upper
                                   : ::oneapi::mkl::uplo::lower;

    ::oneapi::mkl::diag unitdiag = (options & AF_MAT_DIAG_UNIT)
                                       ? ::oneapi::mkl::diag::unit
                                       : ::oneapi::mkl::diag::nonunit;

    buffer<compute_t<T>> aBuf = A.template getBufferWithOffset<compute_t<T>>();
    buffer<compute_t<T>> bBuf = B.template getBufferWithOffset<compute_t<T>>();

    ::oneapi::mkl::blas::trsm(getQueue(), ::oneapi::mkl::side::left, uplo,
                              ::oneapi::mkl::transpose::nontrans, unitdiag,
                              B.dims()[0], B.dims()[1], alpha, aBuf,
                              A.strides()[1], bBuf, B.strides()[1]);
    return B;
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    if (options & AF_MAT_UPPER || options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }

    if (a.dims()[0] == a.dims()[1]) {
        return generalSolve<T>(a, b);
    } else {
        return leastSquares<T>(a, b);
    }
}

#define INSTANTIATE_SOLVE(T)                                                 \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,         \
                               const af_mat_prop options);                   \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b,                          \
                                 const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)
}  // namespace oneapi
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    AF_ERROR("Linear Algebra is disabled on OneAPI", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                 \
    template Array<T> solve<T>(const Array<T> &a, const Array<T> &b,         \
                               const af_mat_prop options);                   \
    template Array<T> solveLU<T>(const Array<T> &A, const Array<int> &pivot, \
                                 const Array<T> &b,                          \
                                 const af_mat_prop options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
