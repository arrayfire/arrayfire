/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <err_cpu.hpp>
#include <handle.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>
#include <cassert>

namespace cpu {

template<typename T>
using gesv_func_def = int (*)(ORDER_TYPE, int, int, T *, int, int *, T *, int);

template<typename T>
using gels_func_def = int (*)(ORDER_TYPE, char, int, int, int, T *, int, T *,
                              int);

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
    A.eval();
    pivot.eval();
    b.eval();

    int N      = A.dims()[0];
    int NRHS   = b.dims()[1];
    Array<T> B = copyArray<T>(b);

    auto func = [=](Param<T> A, Param<T> B, Param<int> pivot, int N, int NRHS) {
        getrs_func<T>()(AF_LAPACK_COL_MAJOR, 'N', N, NRHS, A.get(),
                        A.strides(1), pivot.get(), B.get(), B.strides(1));
    };
    getQueue().enqueue(func, A, B, pivot, N, NRHS);

    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    A.eval();
    b.eval();

    Array<T> B = copyArray<T>(b);
    int N      = B.dims()[0];
    int NRHS   = B.dims()[1];

    auto func = [=](Param<T> A, Param<T> B, int N, int NRHS,
                    const af_mat_prop options) {
        trtrs_func<T>()(AF_LAPACK_COL_MAJOR, options & AF_MAT_UPPER ? 'U' : 'L',
                        'N',  // transpose flag
                        options & AF_MAT_DIAG_UNIT ? 'U' : 'N', N, NRHS,
                        A.get(), A.strides(1), B.get(), B.strides(1));
    };
    getQueue().enqueue(func, A, B, N, NRHS, options);

    return B;
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    a.eval();
    b.eval();

    if (options & AF_MAT_UPPER || options & AF_MAT_LOWER) {
        return triangleSolve<T>(a, b, options);
    }

    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> A = copyArray<T>(a);
    Array<T> B = padArray<T, T>(b, dim4(max(M, N), K));

    if (M == N) {
        Array<int> pivot = createEmptyArray<int>(dim4(N, 1, 1));

        auto func = [=](Param<T> A, Param<T> B, Param<int> pivot, int N,
                        int K) {
            gesv_func<T>()(AF_LAPACK_COL_MAJOR, N, K, A.get(), A.strides(1),
                           pivot.get(), B.get(), B.strides(1));
        };
        getQueue().enqueue(func, A, B, pivot, N, K);
    } else {
        auto func = [=](Param<T> A, Param<T> B, int M, int N, int K) {
            int sM = A.strides(1);
            int sN = A.strides(2) / sM;

            gels_func<T>()(AF_LAPACK_COL_MAJOR, 'N', M, N, K, A.get(),
                           A.strides(1), B.get(), max(sM, sN));
        };
        B.resetDims(dim4(N, K));
        getQueue().enqueue(func, A, B, M, N, K);
    }

    return B;
}

}  // namespace cpu

#else  // WITH_LINEAR_ALGEBRA

namespace cpu {

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options) {
    AF_ERROR("Linear Algebra is disabled on CPU", AF_ERR_NOT_CONFIGURED);
}

}  // namespace cpu

#endif  // WITH_LINEAR_ALGEBRA

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
