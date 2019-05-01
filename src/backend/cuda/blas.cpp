/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <platform.hpp>

#include <arith.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <cublas.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <reduce.hpp>
#include <cassert>
#include <stdexcept>
#include <string>

namespace cuda {

cublasOperation_t toCblasTranspose(af_mat_prop opt) {
    cublasOperation_t out = CUBLAS_OP_N;
    switch (opt) {
        case AF_MAT_NONE: out = CUBLAS_OP_N; break;
        case AF_MAT_TRANS: out = CUBLAS_OP_T; break;
        case AF_MAT_CTRANS: out = CUBLAS_OP_C; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

template<typename T>
struct gemm_func_def_t {
    typedef cublasStatus_t (*gemm_func_def)(cublasHandle_t, cublasOperation_t,
                                            cublasOperation_t, int, int, int,
                                            const T *, const T *, int,
                                            const T *, int, const T *, T *,
                                            int);
};

template<typename T>
struct gemmBatched_func_def_t {
    typedef cublasStatus_t (*gemmBatched_func_def)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const T *, const T **, int, const T **, int, const T *, T **, int, int);
};

template<typename T>
struct gemv_func_def_t {
    typedef cublasStatus_t (*gemv_func_def)(cublasHandle_t, cublasOperation_t,
                                            int, int, const T *, const T *, int,
                                            const T *, int, const T *, T *,
                                            int);
};

template<typename T>
struct trsm_func_def_t {
    typedef cublasStatus_t (*trsm_func_def)(cublasHandle_t, cublasSideMode_t,
                                            cublasFillMode_t, cublasOperation_t,
                                            cublasDiagType_t, int, int,
                                            const T *, const T *, int, T *,
                                            int);
};

#define BLAS_FUNC_DEF(FUNC) \
    template<typename T>    \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, PREFIX)                                       \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cublas##PREFIX##FUNC;                                        \
    }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float, S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble, Z)

BLAS_FUNC_DEF(gemmBatched)
BLAS_FUNC(gemmBatched, float, S)
BLAS_FUNC(gemmBatched, cfloat, C)
BLAS_FUNC(gemmBatched, double, D)
BLAS_FUNC(gemmBatched, cdouble, Z)

BLAS_FUNC_DEF(gemv)
BLAS_FUNC(gemv, float, S)
BLAS_FUNC(gemv, cfloat, C)
BLAS_FUNC(gemv, double, D)
BLAS_FUNC(gemv, cdouble, Z)

BLAS_FUNC_DEF(trsm)
BLAS_FUNC(trsm, float, S)
BLAS_FUNC(trsm, cfloat, C)
BLAS_FUNC(trsm, double, D)
BLAS_FUNC(trsm, cdouble, Z)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

template<typename T, bool conjugate>
struct dot_func_def_t {
    typedef cublasStatus_t (*dot_func_def)(cublasHandle_t, int, const T *, int,
                                           const T *, int, T *);
};

#define BLAS_FUNC_DEF(FUNC)              \
    template<typename T, bool conjugate> \
    typename FUNC##_func_def_t<T, conjugate>::FUNC##_func_def FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX)                       \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               cublas##PREFIX##FUNC;                                   \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, float, true, S)
BLAS_FUNC(dot, double, true, D)
BLAS_FUNC(dot, float, false, S)
BLAS_FUNC(dot, double, false, D)

#undef BLAS_FUNC

#define BLAS_FUNC(FUNC, TYPE, CONJUGATE, PREFIX, SUFFIX)               \
    template<>                                                         \
    typename FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def       \
        FUNC##_func<TYPE, CONJUGATE>() {                               \
        return (FUNC##_func_def_t<TYPE, CONJUGATE>::FUNC##_func_def) & \
               cublas##PREFIX##FUNC##SUFFIX;                           \
    }

BLAS_FUNC_DEF(dot)
BLAS_FUNC(dot, cfloat, true, C, c)
BLAS_FUNC(dot, cdouble, true, Z, c)
BLAS_FUNC(dot, cfloat, false, C, u)
BLAS_FUNC(dot, cdouble, false, Z, u)

#undef BLAS_FUNC
#undef BLAS_FUNC_DEF

using std::max;
using std::vector;

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
                af_mat_prop optRhs) {
    cublasOperation_t lOpts = toCblasTranspose(optLhs);
    cublasOperation_t rOpts = toCblasTranspose(optRhs);

    int aRowDim = (lOpts == CUBLAS_OP_N) ? 0 : 1;
    int aColDim = (lOpts == CUBLAS_OP_N) ? 1 : 0;
    int bColDim = (rOpts == CUBLAS_OP_N) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = lDims[aRowDim];
    int N      = rDims[bColDim];
    int K      = lDims[aColDim];

    dim_t d2     = std::max(lDims[2], rDims[2]);
    dim_t d3     = std::max(lDims[3], rDims[3]);
    dim4 oDims   = dim4(M, N, d2, d3);
    Array<T> out = createEmptyArray<T>(oDims);

    T alpha = scalar<T>(1);
    T beta  = scalar<T>(0);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    dim4 oStrides = out.strides();

    if (oDims.ndims() <= 2) {
        if (rDims[bColDim] == 1) {
            dim_t incr = (optRhs == AF_MAT_NONE) ? rStrides[0] : rStrides[1];
            N          = lDims[aColDim];
            CUBLAS_CHECK(gemv_func<T>()(blasHandle(), lOpts, lDims[0], lDims[1],
                                        &alpha, lhs.get(), lStrides[1],
                                        rhs.get(), incr, &beta, out.get(), 1));
        } else {
            CUBLAS_CHECK(gemm_func<T>()(blasHandle(), lOpts, rOpts, M, N, K,
                                        &alpha, lhs.get(), lStrides[1],
                                        rhs.get(), rStrides[1], &beta,
                                        out.get(), oDims[0]));
        }
    } else {
        int batchSize = oDims[2] * oDims[3];
        std::vector<const T *> lptrs(batchSize);
        std::vector<const T *> rptrs(batchSize);
        std::vector<T *> optrs(batchSize);

        bool is_l_d2_batched = oDims[2] == lDims[2];
        bool is_l_d3_batched = oDims[3] == lDims[3];

        bool is_r_d2_batched = oDims[2] == rDims[2];
        bool is_r_d3_batched = oDims[3] == rDims[3];

        const T *lptr = lhs.get();
        const T *rptr = rhs.get();
        T *optr       = out.get();

        for (int n = 0; n < batchSize; n++) {
            int w    = n / oDims[2];
            int z    = n - w * oDims[2];
            int loff = z * (is_l_d2_batched * lStrides[2]) +
                       w * (is_l_d3_batched * lStrides[3]);
            int roff = z * (is_r_d2_batched * rStrides[2]) +
                       w * (is_r_d3_batched * rStrides[3]);
            lptrs[n] = lptr + loff;
            rptrs[n] = rptr + roff;
            optrs[n] = optr + z * oStrides[2] + w * oStrides[3];
        }

        size_t bytes = batchSize * sizeof(T **);
        auto d_lptrs = memAlloc<uchar>(bytes);
        auto d_rptrs = memAlloc<uchar>(bytes);
        auto d_optrs = memAlloc<uchar>(bytes);
        CUDA_CHECK(cudaMemcpyAsync(d_lptrs.get(), lptrs.data(), bytes,
                                   cudaMemcpyHostToDevice, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(d_rptrs.get(), rptrs.data(), bytes,
                                   cudaMemcpyHostToDevice, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(d_optrs.get(), optrs.data(), bytes,
                                   cudaMemcpyHostToDevice, getActiveStream()));

        // Call this before the gemm call so that you don't have to wait for the
        // computation. Even though it would make more sense to put it
        // afterwards
        CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

        CUBLAS_CHECK(gemmBatched_func<T>()(
            blasHandle(), lOpts, rOpts, M, N, K, &alpha,
            (const T **)d_lptrs.get(), lStrides[1], (const T **)d_rptrs.get(),
            rStrides[1], &beta, (T **)d_optrs.get(), oStrides[1], batchSize));
    }

    return out;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    const Array<T> lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    const Array<T> rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));

    const Array<T> temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
    return reduce<af_add_t, T, T>(temp, 0, false, 0);
}

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, af_mat_prop trans, bool is_upper,
          bool is_left, bool is_unit) {
    // dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = rDims[0];
    int N      = rDims[1];

    T alpha = scalar<T>(1);

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();

    CUBLAS_CHECK(trsm_func<T>()(
        blasHandle(), is_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
        is_upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
        toCblasTranspose(trans),
        is_unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, M, N, &alpha,
        lhs.get(), lStrides[1], rhs.get(), rStrides[1]));
}

#define INSTANTIATE_BLAS(TYPE)                                \
    template Array<TYPE> matmul<TYPE>(const Array<TYPE> &lhs, \
                                      const Array<TYPE> &rhs, \
                                      af_mat_prop optLhs, af_mat_prop optRhs);

INSTANTIATE_BLAS(float)
INSTANTIATE_BLAS(cfloat)
INSTANTIATE_BLAS(double)
INSTANTIATE_BLAS(cdouble)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)

#define INSTANTIATE_TRSM(TYPE)                                               \
    template void trsm<TYPE>(const Array<TYPE> &lhs, Array<TYPE> &rhs,       \
                             af_mat_prop trans, bool is_upper, bool is_left, \
                             bool is_unit);

INSTANTIATE_TRSM(float)
INSTANTIATE_TRSM(cfloat)
INSTANTIATE_TRSM(double)
INSTANTIATE_TRSM(cdouble)

}  // namespace cuda
