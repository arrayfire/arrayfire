/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <copy.hpp>
#include <cublas_v2.h>
#include <cusolverDn.hpp>
#include <identity.hpp>
#include <kernel/triangle.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {

// cusolverStatus_t cusolverDn<>geqrf_bufferSize(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A,
//        int lda,
//        int *Lwork );
//
// cusolverStatus_t cusolverDn<>geqrf(
//        cusolverDnHandle_t handle,
//        int m, int n,
//        <> *A, int lda,
//        <> *TAU,
//        <> *Workspace,
//        int Lwork, int *devInfo );
//
// cusolverStatus_t cusolverDn<>mqr(
//        cusolverDnHandle_t handle,
//        cublasSideMode_t side, cublasOperation_t trans,
//        int m, int n, int k,
//        const double *A, int lda,
//        const double *tau,
//        double *C, int ldc,
//        double *work,
//        int lwork, int *devInfo);

template<typename T>
struct geqrf_func_def_t {
    using geqrf_func_def = cusolverStatus_t (*)(cusolverDnHandle_t, int, int,
                                                T *, int, T *, T *, int, int *);
};

template<typename T>
struct geqrf_buf_func_def_t {
    using geqrf_buf_func_def = cusolverStatus_t (*)(cusolverDnHandle_t, int,
                                                    int, T *, int, int *);
};

template<typename T>
struct mqr_func_def_t {
    using mqr_func_def = cusolverStatus_t (*)(cusolverDnHandle_t,
                                              cublasSideMode_t,
                                              cublasOperation_t, int, int, int,
                                              const T *, int, const T *, T *,
                                              int, T *, int, int *);
};

template<typename T>
struct mqr_buf_func_def_t {
    using mqr_buf_func_def = cusolverStatus_t (*)(cusolverDnHandle_t,
                                                  cublasSideMode_t,
                                                  cublasOperation_t, int, int, int,
                                                  const T *, int, const T *, T *,
                                                  int, int *);
};


#define QR_FUNC_DEF(FUNC)                                         \
    template<typename T>                                          \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func(); \
                                                                  \
    template<typename T>                                          \
    typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def FUNC##_buf_func();

#define QR_FUNC(FUNC, TYPE, PREFIX)                                         \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cusolverDn##PREFIX##FUNC;                                    \
    }                                                                       \
                                                                            \
    template<>                                                              \
    typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def               \
        FUNC##_buf_func<TYPE>() {                                           \
        return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def) &         \
               cusolverDn##PREFIX##FUNC##_bufferSize;                       \
    }

QR_FUNC_DEF(geqrf)
QR_FUNC(geqrf, float, S)
QR_FUNC(geqrf, double, D)
QR_FUNC(geqrf, cfloat, C)
QR_FUNC(geqrf, cdouble, Z)

#define MQR_FUNC_DEF(FUNC)                                        \
    template<typename T>                                          \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func(); \
                                                                  \
    template<typename T>                                          \
    typename FUNC##_buf_func_def_t<T>::FUNC##_buf_func_def FUNC##_buf_func();

#define MQR_FUNC(FUNC, TYPE, PREFIX)                                        \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cusolverDn##PREFIX;                                          \
    }                                                                       \
                                                                            \
    template<>                                                              \
    typename FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def               \
        FUNC##_buf_func<TYPE>() {                                           \
        return (FUNC##_buf_func_def_t<TYPE>::FUNC##_buf_func_def) &         \
               cusolverDn##PREFIX##_bufferSize;                             \
    }

MQR_FUNC_DEF(mqr)
MQR_FUNC(mqr, float, Sormqr)
MQR_FUNC(mqr, double, Dormqr)
MQR_FUNC(mqr, cfloat, Cunmqr)
MQR_FUNC(mqr, cdouble, Zunmqr)

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    Array<T> in_copy = copyArray<T>(in);

    int lwork = 0;

    CUSOLVER_CHECK(geqrf_buf_func<T>()(solverDnHandle(), M, N, in_copy.get(),
                                       in_copy.strides()[1], &lwork));

    auto workspace = memAlloc<T>(lwork);

    t         = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
    auto info = memAlloc<int>(1);

    CUSOLVER_CHECK(geqrf_func<T>()(solverDnHandle(), M, N, in_copy.get(),
                                   in_copy.strides()[1], t.get(),
                                   workspace.get(), lwork, info.get()));

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    kernel::triangle<T>(r, in_copy, true, false);

    int mn = max(M, N);
    dim4 qdims(M, mn);
    q = identity<T>(qdims);

    CUSOLVER_CHECK(mqr_buf_func<T>()(
        solverDnHandle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_N, q.dims()[0],
	q.dims()[1], min(M, N), in_copy.get(), in_copy.strides()[1], t.get(),
        q.get(), q.strides()[1], &lwork));

    workspace = memAlloc<T>(lwork);

    CUSOLVER_CHECK(mqr_func<T>()(
        solverDnHandle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_N, q.dims()[0],
        q.dims()[1], min(M, N), in_copy.get(), in_copy.strides()[1], t.get(),
        q.get(), q.strides()[1], workspace.get(), lwork, info.get()));

    q.resetDims(dim4(M, M));
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    int lwork = 0;

    CUSOLVER_CHECK(geqrf_buf_func<T>()(solverDnHandle(), M, N, in.get(),
                                       in.strides()[1], &lwork));

    auto workspace = memAlloc<T>(lwork);
    auto info      = memAlloc<int>(1);

    CUSOLVER_CHECK(geqrf_func<T>()(solverDnHandle(), M, N, in.get(),
                                   in.strides()[1], t.get(), workspace.get(),
                                   lwork, info.get()));

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
}  // namespace cuda
}  // namespace arrayfire
