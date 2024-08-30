/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <solve.hpp>

#include <blas.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <cublas.hpp>
#include <cusolverDn.hpp>
#include <err_cuda.hpp>
#include <identity.hpp>
#include <lu.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <qr.hpp>
#include <transpose.hpp>

namespace arrayfire {
namespace cuda {

// cublasStatus_t cublas<>getrsBatched( cublasHandle_t handle,
//                                      cublasOperation_t trans,
//                                      int n,
//                                      int nrhs,
//                                      const <> *Aarray[],
//                                      int lda,
//                                      const int *devIpiv,
//                                      <> *Barray[],
//                                      int ldb,
//                                      int *info,
//                                      int batchSize);

template<typename T>
struct getrsBatched_func_def_t {
    typedef cublasStatus_t (*getrsBatched_func_def)(cublasHandle_t,
                                                    cublasOperation_t, int, int,
                                                    const T **, int,
                                                    const int *, T **, int,
                                                    int *, int);
};

// cublasStatus_t cublas<>getrfBatched(cublasHandle_t handle,
//                                     int n,
//                                     float *A[],
//                                     int lda,
//                                     int *P,
//                                     int *info,
//                                     int batchSize);

template<typename T>
struct getrfBatched_func_def_t {
    typedef cublasStatus_t (*getrfBatched_func_def)(cublasHandle_t, int, T **,
                                                    int, int *, int *, int);
};

#define SOLVE_BATCH_FUNC_DEF(FUNC) \
    template<typename T>           \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();

#define SOLVE_BATCH_FUNC(FUNC, TYPE, PREFIX)                                \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cublas##PREFIX##FUNC;                                        \
    }

SOLVE_BATCH_FUNC_DEF(getrfBatched)
SOLVE_BATCH_FUNC(getrfBatched, float, S)
SOLVE_BATCH_FUNC(getrfBatched, double, D)
SOLVE_BATCH_FUNC(getrfBatched, cfloat, C)
SOLVE_BATCH_FUNC(getrfBatched, cdouble, Z)

SOLVE_BATCH_FUNC_DEF(getrsBatched)
SOLVE_BATCH_FUNC(getrsBatched, float, S)
SOLVE_BATCH_FUNC(getrsBatched, double, D)
SOLVE_BATCH_FUNC(getrsBatched, cfloat, C)
SOLVE_BATCH_FUNC(getrsBatched, cdouble, Z)

// cusolverStatus_t cusolverDn<>getrs(
//    cusolverDnHandle_t handle,
//    cublasOperation_t trans,
//    int n, int nrhs,
//    const <> *A, int lda,
//    const int *devIpiv,
//    <> *B, int ldb,
//    int *devInfo );

template<typename T>
struct getrs_func_def_t {
    typedef cusolverStatus_t (*getrs_func_def)(cusolverDnHandle_t,
                                               cublasOperation_t, int, int,
                                               const T *, int, const int *, T *,
                                               int, int *);
};

#define SOLVE_FUNC_DEF(FUNC) \
    template<typename T>     \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();

#define SOLVE_FUNC(FUNC, TYPE, PREFIX)                                      \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cusolverDn##PREFIX##FUNC;                                    \
    }

SOLVE_FUNC_DEF(getrs)
SOLVE_FUNC(getrs, float, S)
SOLVE_FUNC(getrs, double, D)
SOLVE_FUNC(getrs, cfloat, C)
SOLVE_FUNC(getrs, cdouble, Z)

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
struct geqrf_solve_func_def_t {
    typedef cusolverStatus_t (*geqrf_solve_func_def)(cusolverDnHandle_t, int,
                                                     int, T *, int, T *, T *,
                                                     int, int *);
};

template<typename T>
struct geqrf_solve_buf_func_def_t {
    typedef cusolverStatus_t (*geqrf_solve_buf_func_def)(cusolverDnHandle_t,
                                                         int, int, T *, int,
                                                         int *);
};

template<typename T>
struct mqr_solve_func_def_t {
    typedef cusolverStatus_t (*mqr_solve_func_def)(
        cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
        const T *, int, const T *, T *, int, T *, int, int *);
};

template<typename T>
struct mqr_solve_buf_func_def_t {
    typedef cusolverStatus_t (*mqr_solve_buf_func_def)(
	cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
        const T *, int, const T *, T *, int, int *);
};

#define QR_FUNC_DEF(FUNC)                                                     \
    template<typename T>                                                      \
    static typename FUNC##_solve_func_def_t<T>::FUNC##_solve_func_def         \
        FUNC##_solve_func();                                                  \
                                                                              \
    template<typename T>                                                      \
    static typename FUNC##_solve_buf_func_def_t<T>::FUNC##_solve_buf_func_def \
        FUNC##_solve_buf_func();

#define QR_FUNC(FUNC, TYPE, PREFIX)                                       \
    template<>                                                            \
    typename FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def         \
        FUNC##_solve_func<TYPE>() {                                       \
        return (FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def) &   \
               cusolverDn##PREFIX##FUNC;                                  \
    }                                                                     \
                                                                          \
    template<>                                                            \
    typename FUNC##_solve_buf_func_def_t<TYPE>::FUNC##_solve_buf_func_def \
        FUNC##_solve_buf_func<TYPE>() {                                   \
        return (FUNC##_solve_buf_func_def_t<                              \
                   TYPE>::FUNC##_solve_buf_func_def) &                    \
               cusolverDn##PREFIX##FUNC##_bufferSize;                     \
    }

QR_FUNC_DEF(geqrf)
QR_FUNC(geqrf, float, S)
QR_FUNC(geqrf, double, D)
QR_FUNC(geqrf, cfloat, C)
QR_FUNC(geqrf, cdouble, Z)

#define MQR_FUNC_DEF(FUNC)                                                    \
    template<typename T>                                                      \
    static typename FUNC##_solve_func_def_t<T>::FUNC##_solve_func_def         \
        FUNC##_solve_func();                                                  \
	                                                                      \
    template<typename T>                                                      \
    static typename FUNC##_solve_buf_func_def_t<T>::FUNC##_solve_buf_func_def \
       	FUNC##_solve_buf_func();

#define MQR_FUNC(FUNC, TYPE, PREFIX)                                            \
    template<>                                                                  \
    typename FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def               \
        FUNC##_solve_func<TYPE>() {                                             \
        return (FUNC##_solve_func_def_t<TYPE>::FUNC##_solve_func_def) &         \
               cusolverDn##PREFIX;                                              \
    }                                                                           \
                                                                                \
    template<>                                                                  \
    typename FUNC##_solve_buf_func_def_t<TYPE>::FUNC##_solve_buf_func_def       \
        FUNC##_solve_buf_func<TYPE>() {                                         \
        return (FUNC##_solve_buf_func_def_t<TYPE>::FUNC##_solve_buf_func_def) & \
               cusolverDn##PREFIX##_bufferSize;                                 \
    }

MQR_FUNC_DEF(mqr)
MQR_FUNC(mqr, float, Sormqr)
MQR_FUNC(mqr, double, Dormqr)
MQR_FUNC(mqr, cfloat, Cunmqr)
MQR_FUNC(mqr, cdouble, Zunmqr)

template<typename T>
Array<T> solveLU(const Array<T> &A, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options) {
    UNUSED(options);
    int N    = A.dims()[0];
    int NRHS = b.dims()[1];

    Array<T> B = copyArray<T>(b);

    auto info = memAlloc<int>(1);

    CUSOLVER_CHECK(getrs_func<T>()(solverDnHandle(), CUBLAS_OP_N, N, NRHS,
                                   A.get(), A.strides()[1], pivot.get(),
                                   B.get(), B.strides()[1], info.get()));

    return B;
}

template<typename T>
Array<T> generalSolveBatched(const Array<T> &a, const Array<T> &b) {
    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);

    dim4 aDims = a.dims();
    int M      = aDims[0];
    int N      = aDims[1];
    int NRHS   = b.dims()[1];

    if (M != N) {
        AF_ERROR("Batched solve requires square matrices", AF_ERR_ARG);
    }

    int batchz = aDims[2];
    int batchw = aDims[3];
    int batch  = batchz * batchw;

    size_t bytes         = batch * sizeof(T *);
    using unique_mem_ptr = std::unique_ptr<char, void (*)(void *)>;

    unique_mem_ptr aBatched_host_mem(pinnedAlloc<char>(bytes),
                                     pinnedFree);
    unique_mem_ptr bBatched_host_mem(pinnedAlloc<char>(bytes),
                                     pinnedFree);

    T *a_ptr               = A.get();
    T *b_ptr               = B.get();
    T **aBatched_host_ptrs = (T **)aBatched_host_mem.get();
    T **bBatched_host_ptrs = (T **)bBatched_host_mem.get();

    for (int i = 0; i < batchw; i++) {
        for (int j = 0; j < batchz; j++) {
            aBatched_host_ptrs[i * batchz + j] =
                a_ptr + j * A.strides()[2] + i * A.strides()[3];
            bBatched_host_ptrs[i * batchz + j] =
                b_ptr + j * B.strides()[2] + i * B.strides()[3];
        }
    }

    unique_mem_ptr aBatched_device_mem(pinnedAlloc<char>(bytes), pinnedFree);
    unique_mem_ptr bBatched_device_mem(pinnedAlloc<char>(bytes), pinnedFree);

    T **aBatched_device_ptrs = (T **)aBatched_device_mem.get();
    T **bBatched_device_ptrs = (T **)bBatched_device_mem.get();

    CUDA_CHECK(cudaMemcpyAsync(aBatched_device_ptrs, aBatched_host_ptrs, bytes,
                               cudaMemcpyHostToDevice,
                               getStream(getActiveDeviceId())));

    // Perform batched LU
    // getrf requires pivot and info to be device pointers
    Array<int> pivots = createEmptyArray<int>(af::dim4(N, batch, 1, 1));
    Array<int> info   = createEmptyArray<int>(af::dim4(batch, 1, 1, 1));

    CUBLAS_CHECK(getrfBatched_func<T>()(blasHandle(), N, aBatched_device_ptrs,
                                        A.strides()[1], pivots.get(),
                                        info.get(), batch));

    CUDA_CHECK(cudaMemcpyAsync(bBatched_device_ptrs, bBatched_host_ptrs, bytes,
                               cudaMemcpyHostToDevice,
                               getStream(getActiveDeviceId())));

    // getrs requires info to be host pointer
    unique_mem_ptr info_host_mem(pinnedAlloc<char>(batch * sizeof(int)),
                                 pinnedFree);
    CUBLAS_CHECK(getrsBatched_func<T>()(
        blasHandle(), CUBLAS_OP_N, N, NRHS, (const T **)aBatched_device_ptrs,
        A.strides()[1], pivots.get(), bBatched_device_ptrs, B.strides()[1],
        (int *)info_host_mem.get(), batch));
    return B;
}

template<typename T>
Array<T> generalSolve(const Array<T> &a, const Array<T> &b) {
    if (a.dims()[2] > 1 || a.dims()[3] > 1) {
        return generalSolveBatched(a, b);
    }

    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

    Array<T> A       = copyArray<T>(a);
    Array<T> B       = copyArray<T>(b);
    Array<int> pivot = lu_inplace(A, false);

    auto info = memAlloc<int>(1);

    CUSOLVER_CHECK(getrs_func<T>()(solverDnHandle(), CUBLAS_OP_N, N, K, A.get(),
                                   A.strides()[1], pivot.get(), B.get(),
                                   B.strides()[1], info.get()));
    return B;
}

template<typename T>
cublasOperation_t trans() {
    return CUBLAS_OP_T;
}
template<>
cublasOperation_t trans<cfloat>() {
    return CUBLAS_OP_C;
}
template<>
cublasOperation_t trans<cdouble>() {
    return CUBLAS_OP_C;
}

template<typename T>
Array<T> leastSquares(const Array<T> &a, const Array<T> &b) {
    int M = a.dims()[0];
    int N = a.dims()[1];
    int K = b.dims()[1];

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

        int lwork = 0;

        // Get workspace needed for QR
        CUSOLVER_CHECK(geqrf_solve_buf_func<T>()(solverDnHandle(), A.dims()[0],
                                                 A.dims()[1], A.get(),
                                                 A.strides()[1], &lwork));

        auto workspace = memAlloc<T>(lwork);
        Array<T> t     = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
        auto info      = memAlloc<int>(1);

        // In place Perform in place QR
        CUSOLVER_CHECK(geqrf_solve_func<T>()(
            solverDnHandle(), A.dims()[0], A.dims()[1], A.get(), A.strides()[1],
            t.get(), workspace.get(), lwork, info.get()));

        // R1 = R(seq(M), seq(M));
        A.resetDims(dim4(M, M));

        // Bt = tri_solve(R1, B);
        B.resetDims(dim4(M, K));
        trsm<T>(A, B, AF_MAT_CTRANS, true, true, false);

        // Bpad = pad(Bt, ..)
        B.resetDims(dim4(N, K));

        // matmul(Q, Bpad)
        CUSOLVER_CHECK(mqr_solve_buf_func<T>()(
            solverDnHandle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_N, B.dims()[0],
    	    B.dims()[1], A.dims()[0], A.get(), A.strides()[1], t.get(), B.get(),
	    B.strides()[1], &lwork));
    
        workspace = memAlloc<T>(lwork);

        CUSOLVER_CHECK(mqr_solve_func<T>()(
            solverDnHandle(), CUBLAS_SIDE_LEFT, CUBLAS_OP_N, B.dims()[0],
            B.dims()[1], A.dims()[0], A.get(), A.strides()[1], t.get(), B.get(),
            B.strides()[1], workspace.get(), lwork, info.get()));

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

        int lwork = 0;

        // Get workspace needed for QR
        CUSOLVER_CHECK(geqrf_solve_buf_func<T>()(solverDnHandle(), A.dims()[0],
                                                 A.dims()[1], A.get(),
                                                 A.strides()[1], &lwork));

        auto workspace = memAlloc<T>(lwork);
        Array<T> t     = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));
        auto info      = memAlloc<int>(1);

        // In place Perform in place QR
        CUSOLVER_CHECK(geqrf_solve_func<T>()(
            solverDnHandle(), A.dims()[0], A.dims()[1], A.get(), A.strides()[1],
            t.get(), workspace.get(), lwork, info.get()));

        // matmul(Q1, B)
        CUSOLVER_CHECK(mqr_solve_buf_func<T>()(
            solverDnHandle(), CUBLAS_SIDE_LEFT, trans<T>(), M, K, N, A.get(),
	    A.strides()[1], t.get(), B.get(), B.strides()[1], &lwork));
    
        workspace = memAlloc<T>(lwork);

        CUSOLVER_CHECK(mqr_solve_func<T>()(
            solverDnHandle(), CUBLAS_SIDE_LEFT, trans<T>(), M, K, N, A.get(),
            A.strides()[1], t.get(), B.get(), B.strides()[1], workspace.get(),
            lwork, info.get()));

        // tri_solve(R1, Bt)
        A.resetDims(dim4(N, N));
        B.resetDims(dim4(N, K));
        trsm(A, B, AF_MAT_NONE, true, true, false);
    }
    return B;
}

template<typename T>
Array<T> triangleSolve(const Array<T> &A, const Array<T> &b,
                       const af_mat_prop options) {
    Array<T> B = copyArray<T>(b);
    trsm(A, B,
         AF_MAT_NONE,  // transpose flag
         options & AF_MAT_UPPER ? true : false,
         true,  // is_left
         options & AF_MAT_DIAG_UNIT ? true : false);
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

}  // namespace cuda
}  // namespace arrayfire
