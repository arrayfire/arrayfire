/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <cpu/cpu_helper.hpp>
#include <cpu/cpu_solve.hpp>
#include <math.hpp>
#if USE_MKL
#include <mkl_version.h>
#endif
#include <algorithm>
#include <vector>

namespace arrayfire {
namespace opencl {
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
struct mkl_type<cl_float2> {
    using type = MKL_Complex8;
};
template<>
struct mkl_type<cl_double2> {
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
    int N    = A.dims()[0];
    int NRHS = b.dims()[1];

    Array<T> B = copyArray<T>(b);

    mapped_ptr<T> aPtr   = A.getMappedPtr();
    mapped_ptr<T> bPtr   = B.getMappedPtr();
    mapped_ptr<int> pPtr = pivot.getMappedPtr();

    getrs_func<T>()(AF_LAPACK_COL_MAJOR, 'N', N, NRHS, aPtr.get(),
                    A.strides()[1], pPtr.get(), bPtr.get(), B.strides()[1]);

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    Array<T> B = copyArray<T>(b);
    int N      = B.dims()[0];
    int NRHS   = B.dims()[1];

    mapped_ptr<T> aPtr = A.getMappedPtr();
    mapped_ptr<T> bPtr = B.getMappedPtr();

    trtrs_func<T>()(AF_LAPACK_COL_MAJOR, options & AF_MAT_UPPER ? 'U' : 'L',
                    'N',  // transpose flag
                    options & AF_MAT_DIAG_UNIT ? 'U' : 'N', N, NRHS, aPtr.get(),
                    A.strides()[1], bPtr.get(), B.strides()[1]);

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

    mapped_ptr<T> aPtr = A.getMappedPtr();
    mapped_ptr<T> bPtr = B.getMappedPtr();

    getrf_batch_strided_func<typename mkl_type<T>::type>()(
        &M, &N, reinterpret_cast<typename mkl_type<T>::type *>(aPtr.get()),
        &lda, &astride, ipiv.data(), &ipivstride, &batches, info.data());

    getrs_batch_strided_func<typename mkl_type<T>::type>()(
        &trans, &M, &K,
        reinterpret_cast<typename mkl_type<T>::type *>(aPtr.get()), &lda,
        &astride, ipiv.data(), &ipivstride,
        reinterpret_cast<typename mkl_type<T>::type *>(bPtr.get()), &ldb,
        &bstride, &batches, info.data());

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

    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> A = copyArray<T>(a);
    dim4 endPadding(max(M, N) - b.dims()[0], K - b.dims()[1], 0, 0);
    Array<T> B = (endPadding == NullShape
                      ? copyArray(b)
                      : padArrayBorders(b, NullShape, endPadding, AF_PAD_ZERO));

    mapped_ptr<T> aPtr = A.getMappedPtr();
    mapped_ptr<T> bPtr = B.getMappedPtr();

    for (int i = 0; i < batchw; i++) {
        for (int j = 0; j < batchz; j++) {
            auto pA = aPtr.get() + A.strides()[2] * j + A.strides()[3] * i;
            auto pB = bPtr.get() + B.strides()[2] * j + B.strides()[3] * i;

            if (M == N) {
                std::vector<int> pivot(N);
                gesv_func<T>()(AF_LAPACK_COL_MAJOR, N, K, pA, A.strides()[1],
                               &pivot.front(), pB, B.strides()[1]);
            } else {
                int sM = a.strides()[1];
                int sN = a.strides()[2] / sM;

                gels_func<T>()(AF_LAPACK_COL_MAJOR, 'N', M, N, K, pA,
                               A.strides()[1], pB, max(sM, sN));
            }
        }
    }
    if (M != N) { B.resetDims(dim4(N, K, B.dims()[2], B.dims()[3])); }

    return B;
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

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
