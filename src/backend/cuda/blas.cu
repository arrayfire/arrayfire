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
#include <type_traits>
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

template<>
gemm_func_def<schar> gemm_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemm_func_def<schar>();
}
template<>
gemmBatched_func_def<schar> gemmBatched_func<schar>() {
    TYPE_ERROR(3, af_dtype::s8);
    return gemmBatched_func_def<schar>();
}

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

template<typename Ti, typename To = Ti>
cublasStatus_t gemmDispatch(BlasHandle handle, cublasOperation_t lOpts,
                            cublasOperation_t rOpts, int M, int N, int K,
                            const To *alpha, const Array<Ti> &lhs, dim_t lStride,
                            const Array<Ti> &rhs, dim_t rStride, const To *beta,
                            Array<To> &out, dim_t oleading) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3 && __CUDACC_VER_MAJOR__ >= 10) {
        return cublasGemmEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs.get(), getType<Ti>(),
            lStride, rhs.get(), getType<Ti>(), rStride, beta, out.get(),
            getType<To>(), out.strides()[1],
            getComputeType<To>(),  // Compute type

            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemm_func<Nt>()(blasHandle(), lOpts, rOpts, M, N, K, (Nt *)alpha,
                               (Nt *)lhs.get(), lStride, (Nt *)rhs.get(),
                               rStride, (Nt *)beta, (Nt *)out.get(), oleading);

#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

template<typename Ti, typename To = Ti>
cublasStatus_t gemmBatchedDispatch(BlasHandle handle, cublasOperation_t lOpts,
                                   cublasOperation_t rOpts, int M, int N, int K,
                                   const To *alpha, const Ti **lptrs,
                                   int lStrides, const Ti **rptrs, int rStrides,
                                   const To *beta, To **optrs, int oStrides,
                                   int batchSize) {
    auto prop = getDeviceProp(getActiveDeviceId());
#if __CUDACC_VER_MAJOR__ >= 10
    if (prop.major > 3) {
        return cublasGemmBatchedEx(
            blasHandle(), lOpts, rOpts, M, N, K, alpha, (const void **)lptrs,
            getType<Ti>(), lStrides, (const void **)rptrs, getType<Ti>(),
            rStrides, beta, (void **)optrs, getType<Ti>(), oStrides, batchSize,
            getComputeType<Ti>(),  // compute type
            // NOTE: When using the CUBLAS_GEMM_DEFAULT_TENSOR_OP algorithm
            // for the cublasGemm*Ex functions, the performance of the
            // fp32 numbers seem to increase dramatically. Their numerical
            // accuracy is also different compared to regular gemm fuctions.
            // The CUBLAS_GEMM_DEFAULT algorithm selection does not experience
            // this change. Does this imply that the TENSOR_OP function
            // performs the computation in fp16 bit even when the compute
            // type is CUDA_R_32F?
            selectGEMMAlgorithm<Ti>());
    } else {
#endif
        using Nt = typename common::kernel_type<Ti>::native;
        return gemmBatched_func<Nt>()(
            blasHandle(), lOpts, rOpts, M, N, K, (const Nt *)alpha,
            (const Nt **)lptrs, lStrides, (const Nt **)rptrs, rStrides,
            (const Nt *)beta, (Nt **)optrs, oStrides, batchSize);
#if __CUDACC_VER_MAJOR__ >= 10
    }
#endif
}

static void getInputBatchStrides(const dim4 &outDims, const dim4 &inDims,
                                 const dim4 &inStrides, dim_t &stride2,
                                 dim_t &stride3) {
    // Convert A/B batch broadcasting into effective batch strides.
    // If the input does not vary in a batch dimension, its effective stride
    // in that dimension is zero.
    stride2 = (outDims[2] == inDims[2]) ? inStrides[2] : 0;
    stride3 = (outDims[3] == inDims[3]) ? inStrides[3] : 0;
}

static bool getStridedBatchStride(dim_t dim2, dim_t dim3, dim_t stride2,
                                  dim_t stride3, dim_t &batchStride) {
    // stride2/stride3 are already effective batch strides.
    // Check whether X(z,w) = X0 + z*stride2 + w*stride3 can be flattened as
    // X_n = X0 + n*batchStride with n = z + w*dim2.

    if (dim2 <= 1 && dim3 <= 1) {
        batchStride = 0;
        return true;
    }

    if (dim3 <= 1) {
        batchStride = stride2;
        return true;
    }

    if (dim2 <= 1) {
        batchStride = stride3;
        return true;
    }

    const dim_t inRowStep       = stride2;
    const dim_t rowBoundaryStep = stride3 - (dim2 - 1) * stride2;

    batchStride = inRowStep;
    return rowBoundaryStep == inRowStep;
}

#if __CUDACC_VER_MAJOR__ >= 10
template<typename Ti, typename To = Ti>
cublasStatus_t gemmStridedBatchedDispatch(
    BlasHandle handle, cublasOperation_t lOpts, cublasOperation_t rOpts, int M,
    int N, int K, const To *alpha, const Array<Ti> &lhs, dim_t lStride,
    dim_t lBatchStride, const Array<Ti> &rhs, dim_t rStride,
    dim_t rBatchStride, const To *beta, Array<To> &out, dim_t oStride,
    dim_t oBatchStride, int batchSize) {
    return cublasGemmStridedBatchedEx(
        blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs.get(), getType<Ti>(),
        lStride, static_cast<long long int>(lBatchStride), rhs.get(),
        getType<Ti>(), rStride, static_cast<long long int>(rBatchStride), beta,
        out.get(), getType<To>(), oStride,
        static_cast<long long int>(oBatchStride), batchSize,
        getComputeType<To>(), selectGEMMAlgorithm<Ti>());
}
#endif

template<typename Ti, typename To>
void gemm(Array<To> &out, af_mat_prop optLhs, af_mat_prop optRhs, const To *alpha,
          const Array<Ti> &lhs, const Array<Ti> &rhs, const To *beta) {
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
        CUBLAS_CHECK((gemmDispatch<Ti, To>(blasHandle(), lOpts, rOpts, M, N, K, alpha,
                                           lhs, lStrides[1], rhs, rStrides[1], beta,
                                           out, oStrides[1])));
    } else {
        const dim_t batchDim2 = oDims[2];
        const dim_t batchDim3 = oDims[3];
        int batchSize         = batchDim2 * batchDim3;

        dim_t lStride2 = 0;
        dim_t lStride3 = 0;
        dim_t rStride2 = 0;
        dim_t rStride3 = 0;

        getInputBatchStrides(oDims, lDims, lStrides, lStride2, lStride3);
        getInputBatchStrides(oDims, rDims, rStrides, rStride2, rStride3);

        const dim_t oStride2 = oStrides[2];
        const dim_t oStride3 = oStrides[3];

        dim_t lBatchStride = 0;
        dim_t rBatchStride = 0;
        dim_t oBatchStride = 0;

        bool can_use_strided_batched = false;

#if __CUDACC_VER_MAJOR__ >= 10
        {
            auto prop = getDeviceProp(getActiveDeviceId());

            const bool l_is_strided = getStridedBatchStride(
                batchDim2, batchDim3, lStride2, lStride3, lBatchStride);

            const bool r_is_strided = getStridedBatchStride(
                batchDim2, batchDim3, rStride2, rStride3, rBatchStride);

            const bool o_is_strided = getStridedBatchStride(
                batchDim2, batchDim3, oStride2, oStride3, oBatchStride);

            can_use_strided_batched =
                prop.major > 3 && is_same<Ti, To>::value && l_is_strided &&
                r_is_strided && o_is_strided;
        }
#endif

        if (can_use_strided_batched) {
#if __CUDACC_VER_MAJOR__ >= 10
            CUBLAS_CHECK((gemmStridedBatchedDispatch<Ti, To>(
                blasHandle(), lOpts, rOpts, M, N, K, alpha, lhs, lStrides[1],
                lBatchStride, rhs, rStrides[1], rBatchStride, beta, out,
                oStrides[1], oBatchStride, batchSize)));
#endif
        } else {
            vector<const Ti *> lptrs(batchSize);
            vector<const Ti *> rptrs(batchSize);
            vector<To *> optrs(batchSize);

            const Ti *lptr = lhs.get();
            const Ti *rptr = rhs.get();
            To *optr       = out.get();

            for (int n = 0; n < batchSize; n++) {
                int w      = n / batchDim2;
                int z      = n - w * batchDim2;
                dim_t loff = z * lStride2 + w * lStride3;
                dim_t roff = z * rStride2 + w * rStride3;
                dim_t ooff = z * oStride2 + w * oStride3;
                lptrs[n]   = lptr + loff;
                rptrs[n]   = rptr + roff;
                optrs[n]   = optr + ooff;
            }

            size_t bytes = batchSize * sizeof(Ti **);
            auto d_lptrs = memAlloc<uchar>(bytes);
            auto d_rptrs = memAlloc<uchar>(bytes);
            auto d_optrs = memAlloc<uchar>(bytes);
            CUDA_CHECK(cudaMemcpyAsync(d_lptrs.get(), lptrs.data(), bytes,
                                       cudaMemcpyHostToDevice,
                                       getActiveStream()));
            CUDA_CHECK(cudaMemcpyAsync(d_rptrs.get(), rptrs.data(), bytes,
                                       cudaMemcpyHostToDevice,
                                       getActiveStream()));
            CUDA_CHECK(cudaMemcpyAsync(d_optrs.get(), optrs.data(), bytes,
                                       cudaMemcpyHostToDevice,
                                       getActiveStream()));

            // Call this before the gemm call so that you don't have to wait for
            // the computation. Even though it would make more sense to put it
            // afterwards
            CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

            CUBLAS_CHECK(gemmBatchedDispatch(
                blasHandle(), lOpts, rOpts, M, N, K, alpha,
                (const Ti **)d_lptrs.get(), lStrides[1],
                (const Ti **)d_rptrs.get(), rStrides[1], beta,
                (To **)d_optrs.get(), oStrides[1], batchSize));
        }
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

#define INSTANTIATE_GEMM(TYPE, OUTTYPE)                                      \
    template void gemm<TYPE>(Array<OUTTYPE> & out, af_mat_prop optLhs,       \
                             af_mat_prop optRhs, const OUTTYPE *alpha,       \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const OUTTYPE *beta);

INSTANTIATE_GEMM(float, float)
INSTANTIATE_GEMM(cfloat, cfloat)
INSTANTIATE_GEMM(double, double)
INSTANTIATE_GEMM(cdouble, cdouble)
INSTANTIATE_GEMM(half, half)
INSTANTIATE_GEMM(schar, float)

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
