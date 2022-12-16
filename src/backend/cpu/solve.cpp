/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <err_cpu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>
#if USE_MKL
#include <mkl_version.h>
#endif
#include <queue.hpp>
#include <af/dim4.hpp>
#include <algorithm>
#include <complex>
#include <vector>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename T>
using gesv_func_def = int (*)(ORDER_TYPE, int, int, T *, int, int *, T *, int);

template<typename T>
using gels_func_def = int (*)(ORDER_TYPE, char, int, int, int, T *, int, T *,
                              int);

#ifdef AF_USE_MKL_BATCH
template<typename T>
using getrf_batch_strided_func_def =
    void (*)(const MKL_INT *m, const MKL_INT *n, T *a, const MKL_INT *lda,
             const MKL_INT *stride_a, MKL_INT *ipiv, const MKL_INT *stride_ipiv,
             const MKL_INT *batch_size, MKL_INT *info);

#if INTEL_MKL_VERSION >= 20210004
template<typename T>
using getrs_batch_strided_func_def = void (*)(
    const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const T *a,
    const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv,
    const MKL_INT *stride_ipiv, T *b, const MKL_INT *ldb,
    const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info);
#else
template<typename T>
using getrs_batch_strided_func_def =
    void (*)(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, T *a,
             const MKL_INT *lda, const MKL_INT *stride_a, MKL_INT *ipiv,
             const MKL_INT *stride_ipiv, T *b, const MKL_INT *ldb,
             const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info);
#endif
#endif

template<typename T>
using getrs_func_def = int (*)(ORDER_TYPE, char, int, int, const T *, int,
                               const int *, T *, int);

template<typename T>
using trtrs_func_def = int (*)(ORDER_TYPE, char, char, char, int, int,
                               const T *, int, T *, int);

#define SOLVE_FUNC_DEF(FUNC) \
    template<typename T>     \
    FUNC##_func_def<T> FUNC##_func();

#define SOLVE_FUNC(FUNC, TYPE, PREFIX)          \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

SOLVE_FUNC_DEF(gesv)
SOLVE_FUNC(gesv, float, s)
SOLVE_FUNC(gesv, double, d)
SOLVE_FUNC(gesv, cfloat, c)
SOLVE_FUNC(gesv, cdouble, z)

SOLVE_FUNC_DEF(gels)
SOLVE_FUNC(gels, float, s)
SOLVE_FUNC(gels, double, d)
SOLVE_FUNC(gels, cfloat, c)
SOLVE_FUNC(gels, cdouble, z)

#ifdef AF_USE_MKL_BATCH

template<typename T>
struct mkl_type {
    using type = T;
};
template<>
struct mkl_type<std::complex<float>> {
    using type = MKL_Complex8;
};
template<>
struct mkl_type<std::complex<double>> {
    using type = MKL_Complex16;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept-type"
template<typename T>
getrf_batch_strided_func_def<T> getrf_batch_strided_func();

template<>
getrf_batch_strided_func_def<float> getrf_batch_strided_func<float>() {
    return &sgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<double> getrf_batch_strided_func<double>() {
    return &dgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<MKL_Complex8>
getrf_batch_strided_func<MKL_Complex8>() {
    return &cgetrf_batch_strided;
}
template<>
getrf_batch_strided_func_def<MKL_Complex16>
getrf_batch_strided_func<MKL_Complex16>() {
    return &zgetrf_batch_strided;
}

template<typename T>
getrs_batch_strided_func_def<T> getrs_batch_strided_func();

template<>
getrs_batch_strided_func_def<float> getrs_batch_strided_func<float>() {
    return &sgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<double> getrs_batch_strided_func<double>() {
    return &dgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<MKL_Complex8>
getrs_batch_strided_func<MKL_Complex8>() {
    return &cgetrs_batch_strided;
}
template<>
getrs_batch_strided_func_def<MKL_Complex16>
getrs_batch_strided_func<MKL_Complex16>() {
    return &zgetrs_batch_strided;
}

#pragma GCC diagnostic pop
#endif

SOLVE_FUNC_DEF(getrs)
SOLVE_FUNC(getrs, float, s)
SOLVE_FUNC(getrs, double, d)
SOLVE_FUNC(getrs, cfloat, c)
SOLVE_FUNC(getrs, cdouble, z)

SOLVE_FUNC_DEF(trtrs)
SOLVE_FUNC(trtrs, float, s)
SOLVE_FUNC(trtrs, double, d)
SOLVE_FUNC(trtrs, cfloat, c)
SOLVE_FUNC(trtrs, cdouble, z)

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    UNUSED(options);
    int N      = A.dims()[0];
    int NRHS   = b.dims()[1];
    Array<T> B = copyArray<T>(b);

    // NOLINTNEXTLINE
    auto func = [=](CParam<T> A, Param<T> B, CParam<int> pivot, int N,
                    int NRHS) {
        getrs_func<T>()(AF_LAPACK_COL_MAJOR, 'N', N, NRHS, A.get(),
                        A.strides(1), pivot.get(), B.get(), B.strides(1));
    };
    getQueue().enqueue(func, A, B, pivot, N, NRHS);

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    Array<T> B = copyArray<T>(b);
    int N      = B.dims()[0];
    int NRHS   = B.dims()[1];

    auto func = [=](const CParam<T> A, Param<T> B, int N, int NRHS,
                    const af_mat_prop options) {
        trtrs_func<T>()(AF_LAPACK_COL_MAJOR, options & AF_MAT_UPPER ? 'U' : 'L',
                        'N',  // transpose flag
                        options & AF_MAT_DIAG_UNIT ? 'U' : 'N', N, NRHS,
                        A.get(), A.strides(1), B.get(), B.strides(1));
    };
    getQueue().enqueue(func, A, B, N, NRHS, options);

    return B;
}

#ifdef AF_USE_MKL_BATCH

template<typename T>
Array<T> generalSolveBatched(const Array<T> &a, const Array<T> &b,
                             const af_mat_prop options) {
    using std::vector;
    int batches = a.dims()[2] * a.dims()[3];

    dim4 aDims = a.dims();
    dim4 bDims = b.dims();
    int M      = aDims[0];
    int N      = aDims[1];
    int K      = bDims[1];
    int MN     = std::min(M, N);

    int lda     = a.strides()[1];
    int astride = a.strides()[2];

    vector<int> ipiv(MN * batches);
    int ipivstride = MN;

    int ldb     = b.strides()[1];
    int bstride = b.strides()[2];

    vector<int> info(batches, 0);

    char trans = 'N';

    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);

    auto getrf_rs = [](char TRANS, int M, int N, int K, Param<T> a, int LDA,
                       int ASTRIDE, vector<int> IPIV, int IPIVSTRIDE,
                       Param<T> b, int LDB, int BSTRIDE, int BATCH_SIZE,
                       vector<int> INFO) {
        getrf_batch_strided_func<typename mkl_type<T>::type>()(
            &M, &N, reinterpret_cast<typename mkl_type<T>::type *>(a.get()),
            &LDA, &ASTRIDE, IPIV.data(), &IPIVSTRIDE, &BATCH_SIZE, INFO.data());

        getrs_batch_strided_func<typename mkl_type<T>::type>()(
            &TRANS, &M, &K,
            reinterpret_cast<typename mkl_type<T>::type *>(a.get()), &LDA,
            &ASTRIDE, IPIV.data(), &IPIVSTRIDE,
            reinterpret_cast<typename mkl_type<T>::type *>(b.get()), &LDB,
            &BSTRIDE, &BATCH_SIZE, INFO.data());
    };

    getQueue().enqueue(getrf_rs, trans, M, N, K, A, lda, astride, ipiv,
                       ipivstride, B, ldb, bstride, batches, info);

    return B;
}
#endif

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    if (options & AF_MAT_UPPER || options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }

#ifdef AF_USE_MKL_BATCH
    if (a.dims()[2] > 1 || a.dims()[3] > 1) {
        return generalSolveBatched(a, b, options);
    }
#endif

    const dim4 NullShape(0, 0, 0, 0);

    dim4 aDims = a.dims();
    int batchz = aDims[2];
    int batchw = aDims[3];

    int M = aDims[0];
    int N = aDims[1];
    int K = b.dims()[1];

    Array<T> A = copyArray<T>(a);

    dim4 endPadding(max(M, N) - b.dims()[0], K - b.dims()[1], 0, 0);
    Array<T> B = (endPadding == NullShape
                      ? copyArray(b)
                      : padArrayBorders(b, NullShape, endPadding, AF_PAD_ZERO));

    for (int i = 0; i < batchw; i++) {
        for (int j = 0; j < batchz; j++) {
            Param<T> pA(A.get() + A.strides()[2] * j + A.strides()[3] * i,
                        A.dims(), A.strides());
            Param<T> pB(B.get() + B.strides()[2] * j + B.strides()[3] * i,
                        B.dims(), B.strides());
            if (M == N) {
                Array<int> pivot = createEmptyArray<int>(dim4(N, 1, 1));

                auto func = [](Param<T> A, Param<T> B, Param<int> pivot, int N,
                               int K) {
                    gesv_func<T>()(AF_LAPACK_COL_MAJOR, N, K, A.get(),
                                   A.strides(1), pivot.get(), B.get(),
                                   B.strides(1));
                };
                getQueue().enqueue(func, pA, pB, pivot, N, K);
            } else {
                auto func = [=](Param<T> A, Param<T> B, int M, int N, int K) {
                    int sM = A.dims(0);
                    int sN = A.dims(1);

                    gels_func<T>()(AF_LAPACK_COL_MAJOR, 'N', M, N, K, A.get(),
                                   A.strides(1), B.get(), max(sM, sN));
                };
                getQueue().enqueue(func, pA, pB, M, N, K);
            }
        }
    }

    if (M != N) { B.resetDims(dim4(N, K, B.dims()[2], B.dims()[3])); }

    return B;
}

}  // namespace cpu
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    AF_ERROR(
        "This version of ArrayFire was built without linear algebra routines",
        AF_ERR_NOT_CONFIGURED);
}

}  // namespace cpu
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace cpu {

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

}  // namespace cpu
}  // namespace arrayfire
