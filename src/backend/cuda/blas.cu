/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cublas.hpp>
#include <cublas_v2.h>
#include <cudaDataType.hpp>
#include <cuda_runtime.h>
#include <err_cuda.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <tile.hpp>
#include <transpose.hpp>
#include <types.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using arrayfire::common::half;
using arrayfire::common::kernel_type;
using std::is_same;
using std::vector;

namespace arrayfire {
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
using gemm_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const T *, const T *, int, const T *, int, const T *, T *, int)>;

template<typename T>
using gemmBatched_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const T *, const T **, int, const T **, int, const T *, T **, int, int)>;

template<typename T>
using trsm_func_def = std::function<cublasStatus_t(
    cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, int, const T *, const T *, int, T *, int)>;

#define BLAS_FUNC_DEF(FUNC) \
    template<typename T>    \
    FUNC##_func_def<T> FUNC##_func();

#define BLAS_FUNC(FUNC, TYPE, PREFIX)           \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &cublas##PREFIX##FUNC;           \
    }

BLAS_FUNC_DEF(gemm)
BLAS_FUNC(gemm, float, S)
BLAS_FUNC(gemm, cfloat, C)
BLAS_FUNC(gemm, double, D)
BLAS_FUNC(gemm, cdouble, Z)
BLAS_FUNC(gemm, __half, H)

BLAS_FUNC_DEF(gemmBatched)
BLAS_FUNC(gemmBatched, float, S)
BLAS_FUNC(gemmBatched, cfloat, C)
BLAS_FUNC(gemmBatched, double, D)
BLAS_FUNC(gemmBatched, cdouble, Z)
BLAS_FUNC(gemmBatched, __half, H)

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

template<typename T>
cublasGemmAlgo_t selectGEMMAlgorithm() {
    return CUBLAS_GEMM_DEFAULT;
}

template<>
cublasGemmAlgo_t selectGEMMAlgorithm<common::half>() {
    auto dev              = getDeviceProp(getActiveDeviceId());
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    if (dev.major >= 7) { algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; }
    return algo;
}

template<>
cublasGemmAlgo_t selectGEMMAlgorithm<__half>() {
    return selectGEMMAlgorithm<common::half>();
}

template<typename T>
cublasStatus_t gemmDispatch(BlasHandle handle, cublasOperation_t lOpts,
                            cublasOperation_t rOpts, int M, int N, int K,
                            const T *alpha, const Array<T> &lhs, dim_t lStride,
                            const Array<T> &rhs, dim_t rStride, const T *beta,
                            Array<T> &out, dim_t oleading) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3 && __CUDACC_VER_MAJOR__ >= 10) {
        return cublasGemmEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs.get(), getType<T>(),
            lStride, rhs.get(), getType<T>(), rStride, beta, out.get(),
            getType<T>(), out.strides()[1],
            getComputeType<T>(),  // Compute type

            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<T>());
    } else {
#endif
        using Nt = typename common::kernel_type<T>::native;
        return gemm_func<Nt>()(blasHandle(), lOpts, rOpts, M, N, K, (Nt *)alpha,
                               (Nt *)lhs.get(), lStride, (Nt *)rhs.get(),
                               rStride, (Nt *)beta, (Nt *)out.get(), oleading);

#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename T>
cublasStatus_t gemmBatchedDispatch(BlasHandle handle, cublasOperation_t lOpts,
                                   cublasOperation_t rOpts, int M, int N, int K,
                                   const T *alpha, const T **lptrs,
                                   int lStrides, const T **rptrs, int rStrides,
                                   const T *beta, T **optrs, int oStrides,
                                   int batchSize) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3) {
        return cublasGemmBatchedEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, (const void **)lptrs,
            getType<T>(), lStrides, (const void **)rptrs, getType<T>(),
            rStrides, beta, (void **)optrs, getType<T>(), oStrides, batchSize,
            getComputeType<T>(),  // compute type
            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<T>());
    } else {
#endif
        using Nt = typename common::kernel_type<T>::native;
        return gemmBatched_func<Nt>()(
            blasHandle(), lOpts, rOpts, M, N, K, (const Nt *)alpha,
            (const Nt **)lptrs, lStrides, (const Nt **)rptrs, rStrides,
            (const Nt *)beta, (Nt **)optrs, oStrides, batchSize);
#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename T>
void gemm(Array<T> &out, af_mat_prop optLhs, af_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta) {
    const cublasOperation_t lOpts = toCblasTranspose(optLhs);
    const cublasOperation_t rOpts = toCblasTranspose(optRhs);

    const int aRowDim = (lOpts == CUBLAS_OP_N) ? 0 : 1;
    const int aColDim = (lOpts == CUBLAS_OP_N) ? 1 : 0;
    const int bColDim = (rOpts == CUBLAS_OP_N) ? 1 : 0;

    const dim4 lDims = lhs.dims();
    const dim4 rDims = rhs.dims();
    const int M      = lDims[aRowDim];
    const int N      = rDims[bColDim];
    const int K      = lDims[aColDim];
    const dim4 oDims = out.dims();

    dim4 lStrides = lhs.strides();
    dim4 rStrides = rhs.strides();
    dim4 oStrides = out.strides();

    if (oDims.ndims() <= 2) {
        CUBLAS_CHECK(gemmDispatch<T>(blasHandle(), lOpts, rOpts, M, N, K, alpha,
                                     lhs, lStrides[1], rhs, rStrides[1], beta,
                                     out, oStrides[1]));
    } else {
        int batchSize = oDims[2] * oDims[3];
        vector<const T *> lptrs(batchSize);
        vector<const T *> rptrs(batchSize);
        vector<T *> optrs(batchSize);

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

        using Nt = typename common::kernel_type<T>::native;
        CUBLAS_CHECK(gemmBatchedDispatch(
            blasHandle(), lOpts, rOpts, M, N, K, alpha,
            (const T **)d_lptrs.get(), lStrides[1], (const T **)d_rptrs.get(),
            rStrides[1], beta, (T **)d_optrs.get(), oStrides[1], batchSize));
    }
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    auto lhs_ = (optLhs == AF_MAT_NONE ? lhs : conj<T>(lhs));
    auto rhs_ = (optRhs == AF_MAT_NONE ? rhs : conj<T>(rhs));
    auto temp = arithOp<T, af_mul_t>(lhs_, rhs_, lhs_.dims());
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

#define INSTANTIATE_GEMM(TYPE)                                               \
    template void gemm<TYPE>(Array<TYPE> & out, af_mat_prop optLhs,          \
                             af_mat_prop optRhs, const TYPE *alpha,          \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const TYPE *beta);

INSTANTIATE_GEMM(float)
INSTANTIATE_GEMM(cfloat)
INSTANTIATE_GEMM(double)
INSTANTIATE_GEMM(cdouble)
INSTANTIATE_GEMM(half)

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs);

INSTANTIATE_DOT(float)
INSTANTIATE_DOT(double)
INSTANTIATE_DOT(cfloat)
INSTANTIATE_DOT(cdouble)
INSTANTIATE_DOT(half)

#define INSTANTIATE_TRSM(TYPE)                                               \
    template void trsm<TYPE>(const Array<TYPE> &lhs, Array<TYPE> &rhs,       \
                             af_mat_prop trans, bool is_upper, bool is_left, \
                             bool is_unit);

INSTANTIATE_TRSM(float)
INSTANTIATE_TRSM(cfloat)
INSTANTIATE_TRSM(double)
INSTANTIATE_TRSM(cdouble)

}  // namespace cuda
}  // namespace arrayfire
