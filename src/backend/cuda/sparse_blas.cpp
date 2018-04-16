/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>
#include <cuda_runtime.h>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <common/err_common.hpp>
#include <math.hpp>
#include <complex.hpp>

namespace cuda
{

using namespace std;

cusparseOperation_t
toCusparseTranspose(af_mat_prop opt)
{
    cusparseOperation_t out = CUSPARSE_OPERATION_NON_TRANSPOSE;
    switch(opt) {
        case AF_MAT_NONE        : out = CUSPARSE_OPERATION_NON_TRANSPOSE;       break;
        case AF_MAT_TRANS       : out = CUSPARSE_OPERATION_TRANSPOSE;           break;
        case AF_MAT_CTRANS      : out = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; break;
        default                 : AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }
    return out;
}

//cusparseStatus_t cusparseZcsrmm(  cusparseHandle_t handle,
//                                  cusparseOperation_t transA,
//                                  int m, int n, int k, int nnz,
//                                  const cuDoubleComplex *alpha,
//                                  const cusparseMatDescr_t descrA,
//                                  const cuDoubleComplex *csrValA,
//                                  const int *csrRowPtrA, const int *csrColIndA,
//                                  const cuDoubleComplex *B, int ldb,
//                                  const cuDoubleComplex *beta,
//                                  cuDoubleComplex *C, int ldc);

template<typename T>
struct csrmm_func_def_t
{
    typedef cusparseStatus_t (*csrmm_func_def)( cusparseHandle_t,
                                                cusparseOperation_t,
                                                int, int, int, int,
                                                const T *,
                                                const cusparseMatDescr_t,
                                                const T *, const int *, const int *,
                                                const T *, int,
                                                const T *,
                                                T *, int);
};

//cusparseStatus_t cusparseZcsrmv(  cusparseHandle_t handle,
//                                  cusparseOperation_t transA,
//                                  int m, int n, int nnz,
//                                  const cuDoubleComplex *alpha,
//                                  const cusparseMatDescr_t descrA,
//                                  const cuDoubleComplex *csrValA,
//                                  const int *csrRowPtrA, const int *csrColIndA,
//                                  const cuDoubleComplex *x,
//                                  const cuDoubleComplex *beta,
//                                  cuDoubleComplex *y)

template<typename T>
struct csrmv_func_def_t
{
    typedef cusparseStatus_t (*csrmv_func_def)( cusparseHandle_t,
                                                cusparseOperation_t,
                                                int, int, int,
                                                const T *,
                                                const cusparseMatDescr_t,
                                                const T *, const int *, const int *,
                                                const T *,
                                                const T *,
                                                T *);
};

//cusparseStatus_t cusparseZcsr2csc(cusparseHandle_t handle,
//                                  int m, int n, int nnz,
//                                  const cuDoubleComplex *csrSortedVal,
//                                  const int *csrSortedRowPtr, const int *csrSortedColInd,
//                                  cuDoubleComplex *cscSortedVal,
//                                  int *cscSortedRowInd, int *cscSortedColPtr,
//                                  cusparseAction_t copyValues,
//                                  cusparseIndexBase_t idxBase);

#define SPARSE_FUNC_DEF( FUNC )                     \
template<typename T>                                \
typename FUNC##_func_def_t<T>::FUNC##_func_def      \
FUNC##_func();

#define SPARSE_FUNC( FUNC, TYPE, PREFIX )                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def        \
FUNC##_func<TYPE>()                                                 \
{ return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cusparse##PREFIX##FUNC; }

SPARSE_FUNC_DEF(csrmm)
SPARSE_FUNC(csrmm, float,  S)
SPARSE_FUNC(csrmm, double, D)
SPARSE_FUNC(csrmm, cfloat, C)
SPARSE_FUNC(csrmm, cdouble,Z)

SPARSE_FUNC_DEF(csrmv)
SPARSE_FUNC(csrmv, float,  S)
SPARSE_FUNC(csrmv, double, D)
SPARSE_FUNC(csrmv, cfloat, C)
SPARSE_FUNC(csrmv, cdouble,Z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                af_mat_prop optLhs, af_mat_prop optRhs)
{
    // Similar Operations to GEMM
    cusparseOperation_t lOpts = toCusparseTranspose(optLhs);

    int lRowDim = (lOpts == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 0 : 1;
    //int lColDim = (lOpts == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;
    static const int rColDim = 1; //Unsupported : (rOpts == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : 0;

    dim4 lDims = lhs.dims();
    dim4 rDims = rhs.dims();
    int M = lDims[lRowDim];
    int N = rDims[rColDim];
    //int K = lDims[lColDim];

    Array<T> out = createEmptyArray<T>(af::dim4(M, N, 1, 1));
    T alpha = scalar<T>(1);
    T beta  = scalar<T>(0);

    dim4 rStrides = rhs.strides();

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
    if(rDims[rColDim] == 1) {
        CUSPARSE_CHECK(csrmv_func<T>()(
                       sparseHandle(),
                       lOpts,
                       lDims[0], lDims[1], lhs.getNNZ(),
                       &alpha,
                       descr, lhs.getValues().get(),
                       lhs.getRowIdx().get(), lhs.getColIdx().get(),
                       rhs.get(),
                       &beta,
                       out.get()));
    } else {
        CUSPARSE_CHECK(csrmm_func<T>()(
                       sparseHandle(),
                       lOpts,
                       lDims[0], rDims[rColDim], lDims[1], lhs.getNNZ(),
                       &alpha,
                       descr, lhs.getValues().get(),
                       lhs.getRowIdx().get(), lhs.getColIdx().get(),
                       rhs.get(), rStrides[1],
                       &beta,
                       out.get(),
                       out.dims()[0]));
    }

    // Destory Sparse Matrix Descriptor
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));

    return out;
}

#define INSTANTIATE_SPARSE(T)                                                           \
    template Array<T> matmul<T>(const common::SparseArray<T> lhs, const Array<T> rhs,   \
                                af_mat_prop optLhs, af_mat_prop optRhs);                \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}
