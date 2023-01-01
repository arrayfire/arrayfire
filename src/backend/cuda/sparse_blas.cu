/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#include <common/err_common.hpp>
#include <complex.hpp>
#include <cudaDataType.hpp>
#include <cuda_runtime.h>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <math.hpp>
#include <platform.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

cusparseOperation_t toCusparseTranspose(af_mat_prop opt) {
    cusparseOperation_t out = CUSPARSE_OPERATION_NON_TRANSPOSE;
    switch (opt) {
        case AF_MAT_NONE: out = CUSPARSE_OPERATION_NON_TRANSPOSE; break;
        case AF_MAT_TRANS: out = CUSPARSE_OPERATION_TRANSPOSE; break;
        case AF_MAT_CTRANS: out = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; break;
        default: AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

#if CUSPARSE_VERSION < 11300
#define AF_CUSPARSE_SPMV_CSR_ALG1 CUSPARSE_CSRMV_ALG1
#define AF_CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_CSRMM_ALG1
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_CSRMM_ALG1
#elif CUSPARSE_VERSION < 11400
#define AF_CUSPARSE_SPMV_CSR_ALG1 CUSPARSE_CSRMV_ALG1
#define AF_CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_SPMM_CSR_ALG1
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_SPMM_CSR_ALG1
#else
#define AF_CUSPARSE_SPMV_CSR_ALG1 CUSPARSE_SPMV_CSR_ALG1
#define AF_CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_SPMV_ALG_DEFAULT
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_SPMM_CSR_ALG1
#define AF_CUSPARSE_SPMM_CSR_ALG1 CUSPARSE_SPMM_CSR_ALG1
#endif

#if defined(AF_USE_NEW_CUSPARSE_API)

template<typename T>
size_t spmvBufferSize(cusparseOperation_t opA, const T *alpha,
                      const cusparseSpMatDescr_t matA,
                      const cusparseDnVecDescr_t vecX, const T *beta,
                      const cusparseDnVecDescr_t vecY) {
    size_t retVal = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        sparseHandle(), opA, alpha, matA, vecX, beta, vecY, getComputeType<T>(),
        AF_CUSPARSE_SPMV_CSR_ALG1, &retVal));
    return retVal;
}

template<typename T>
void spmv(cusparseOperation_t opA, const T *alpha,
          const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX,
          const T *beta, const cusparseDnVecDescr_t vecY, void *buffer) {
    CUSPARSE_CHECK(cusparseSpMV(sparseHandle(), opA, alpha, matA, vecX, beta,
                                vecY, getComputeType<T>(),
                                AF_CUSPARSE_SPMV_ALG_DEFAULT, buffer));
}

template<typename T>
size_t spmmBufferSize(cusparseOperation_t opA, cusparseOperation_t opB,
                      const T *alpha, const cusparseSpMatDescr_t matA,
                      const cusparseDnMatDescr_t matB, const T *beta,
                      const cusparseDnMatDescr_t matC) {
    size_t retVal = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sparseHandle(), opA, opB, alpha, matA, matB, beta, matC,
        getComputeType<T>(), AF_CUSPARSE_SPMM_CSR_ALG1, &retVal));
    return retVal;
}

template<typename T>
void spmm(cusparseOperation_t opA, cusparseOperation_t opB, const T *alpha,
          const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB,
          const T *beta, const cusparseDnMatDescr_t matC, void *buffer) {
    CUSPARSE_CHECK(cusparseSpMM(sparseHandle(), opA, opB, alpha, matA, matB,
                                beta, matC, getComputeType<T>(),
                                AF_CUSPARSE_SPMM_CSR_ALG1, buffer));
}

#else

template<typename T>
struct csrmv_func_def_t {
    typedef cusparseStatus_t (*csrmv_func_def)(
        cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
        int k, const T *alpha, const cusparseMatDescr_t descrA,
        const T *csrValA, const int *csrRowPtrA, const int *csrColIndA,
        const T *x, const T *beta, T *y);
};

template<typename T>
struct csrmm_func_def_t {
    typedef cusparseStatus_t (*csrmm_func_def)(
        cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
        int k, int nnz, const T *alpha, const cusparseMatDescr_t descrA,
        const T *csrValA, const int *csrRowPtrA, const int *csrColIndA,
        const T *B, int ldb, const T *beta, T *C, int ldc);
};

#define SPARSE_FUNC_DEF(FUNC) \
    template<typename T>      \
    typename FUNC##_func_def_t<T>::FUNC##_func_def FUNC##_func();

#define SPARSE_FUNC(FUNC, TYPE, PREFIX)                                     \
    template<>                                                              \
    typename FUNC##_func_def_t<TYPE>::FUNC##_func_def FUNC##_func<TYPE>() { \
        return (FUNC##_func_def_t<TYPE>::FUNC##_func_def) &                 \
               cusparse##PREFIX##FUNC;                                      \
    }

SPARSE_FUNC_DEF(csrmm)
SPARSE_FUNC(csrmm, float, S)
SPARSE_FUNC(csrmm, double, D)
SPARSE_FUNC(csrmm, cfloat, C)
SPARSE_FUNC(csrmm, cdouble, Z)

SPARSE_FUNC_DEF(csrmv)
SPARSE_FUNC(csrmv, float, S)
SPARSE_FUNC(csrmv, double, D)
SPARSE_FUNC(csrmv, cfloat, C)
SPARSE_FUNC(csrmv, cdouble, Z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

#endif

template<typename T>
Array<T> matmul(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs) {
    // Similar Operations to GEMM
    cusparseOperation_t lOpts = toCusparseTranspose(optLhs);

    int lRowDim = (lOpts == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    // int lColDim = (lOpts == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;
    static const int rColDim = 1;  // Unsupported : (rOpts ==
                                   // CUSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M      = lDims[lRowDim];
    int N      = rDims[rColDim];
    // int K = lDims[lColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    T alpha      = scalar<T>(1);
    T beta       = scalar<T>(0);

    dim4 rStrides = rhs.strides();

#if defined(AF_USE_NEW_CUSPARSE_API)

    auto spMat = cusparseDescriptor<T>(lhs);

    if (rDims[rColDim] == 1) {
        auto dnVec = denVecDescriptor<T>(rhs);
        auto dnOut = denVecDescriptor<T>(out);
        size_t bufferSize =
            spmvBufferSize<T>(lOpts, &alpha, spMat, dnVec, &beta, dnOut);
        auto tempBuffer = createEmptyArray<char>(dim4(bufferSize));
        spmv<T>(lOpts, &alpha, spMat, dnVec, &beta, dnOut, tempBuffer.get());
    } else {
        cusparseOperation_t rOpts = toCusparseTranspose(optRhs);

        auto dnMat = denMatDescriptor<T>(rhs);
        auto dnOut = denMatDescriptor<T>(out);
        size_t bufferSize =
            spmmBufferSize<T>(lOpts, rOpts, &alpha, spMat, dnMat, &beta, dnOut);
        auto tempBuffer = createEmptyArray<char>(dim4(bufferSize));
        spmm<T>(lOpts, rOpts, &alpha, spMat, dnMat, &beta, dnOut,
                tempBuffer.get());
    }

#else

    // Create Sparse Matrix Descriptor
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    CUSPARSE_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // Call Matrix-Vector or Matrix-Matrix
    // Note:
    // Do not use M, N, K here. Use lDims and rDims instead.
    // This is because the function wants row/col of A
    // and not OP(A) (gemm wants row/col of OP(A)).
    if (rDims[rColDim] == 1) {
        CUSPARSE_CHECK(csrmv_func<T>()(
            sparseHandle(), lOpts, lDims[0], lDims[1], lhs.getNNZ(), &alpha,
            descr, lhs.getValues().get(), lhs.getRowIdx().get(),
            lhs.getColIdx().get(), rhs.get(), &beta, out.get()));
    } else {
        CUSPARSE_CHECK(csrmm_func<T>()(
            sparseHandle(), lOpts, lDims[0], rDims[rColDim], lDims[1],
            lhs.getNNZ(), &alpha, descr, lhs.getValues().get(),
            lhs.getRowIdx().get(), lhs.getColIdx().get(), rhs.get(),
            rStrides[1], &beta, out.get(), out.dims()[0]));
    }
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));

#endif

    return out;
}

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T> &lhs,       \
                                const Array<T> &rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace cuda
}  // namespace arrayfire
