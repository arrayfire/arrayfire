/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse.hpp>
#include <cusparseManager.hpp>
#include <cuda_runtime.h>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <err_common.hpp>
#include <math.hpp>
#include <complex.hpp>

namespace cuda
{

using cusparse::getHandle;
using namespace std;

//cusparseStatus_t cusparseZcsr2csc(cusparseHandle_t handle,
//                                  int m, int n, int nnz,
//                                  const cuDoubleComplex *csrSortedVal,
//                                  const int *csrSortedRowPtr, const int *csrSortedColInd,
//                                  cuDoubleComplex *cscSortedVal,
//                                  int *cscSortedRowInd, int *cscSortedColPtr,
//                                  cusparseAction_t copyValues,
//                                  cusparseIndexBase_t idxBase);

template<typename T>
struct csr2csc_func_def_t
{
    typedef cusparseStatus_t (*csr2csc_func_def)( cusparseHandle_t,
                                                  int, int, int,
                                                  const T *, const int *, const int *,
                                                  T *, int *, int *,
                                                  cusparseAction_t,
                                                  cusparseIndexBase_t);
};

//cusparseStatus_t cusparseZdense2csr(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *A, int lda,
//                                    const int *nnzPerRow,
//                                    cuDoubleComplex *csrValA,
//                                    int *csrRowPtrA, int *csrColIndA)
template<typename T>
struct dense2csr_func_def_t
{
    typedef cusparseStatus_t (*dense2csr_func_def)( cusparseHandle_t,
                                                    int, int,
                                                    const cusparseMatDescr_t,
                                                    const T *, int,
                                                    const int *,
                                                    T *,
                                                    int *, int *);
};

//cusparseStatus_t cusparseZdense2csc(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *A, int lda,
//                                    const int *nnzPerCol,
//                                    cuDoubleComplex *cscValA,
//                                    int *cscRowIndA, int *cscColPtrA)
template<typename T>
struct dense2csc_func_def_t
{
    typedef cusparseStatus_t (*dense2csc_func_def)( cusparseHandle_t,
                                                    int, int,
                                                    const cusparseMatDescr_t,
                                                    const T *, int,
                                                    const int *,
                                                    T *,
                                                    int *, int *);
};

//cusparseStatus_t cusparseZnnz(cusparseHandle_t handle,
//                              cusparseDirection_t dirA,
//                              int m, int n,
//                              const cusparseMatDescr_t descrA,
//                              const cuDoubleComplex *A, int lda,
//                              int *nnzPerRowColumn,
//                              int *nnzTotalDevHostPtr)
template<typename T>
struct nnz_func_def_t
{
    typedef cusparseStatus_t (*nnz_func_def)( cusparseHandle_t,
                                              cusparseDirection_t,
                                              int, int,
                                              const cusparseMatDescr_t,
                                              const T *, int,
                                              int *, int *);
};

#define SPARSE_FUNC_DEF( FUNC )                     \
template<typename T>                                \
typename FUNC##_func_def_t<T>::FUNC##_func_def      \
FUNC##_func();

#define SPARSE_FUNC( FUNC, TYPE, PREFIX )                           \
template<> typename FUNC##_func_def_t<TYPE>::FUNC##_func_def        \
FUNC##_func<TYPE>()                                                 \
{ return (FUNC##_func_def_t<TYPE>::FUNC##_func_def)&cusparse##PREFIX##FUNC; }

SPARSE_FUNC_DEF(csr2csc)
SPARSE_FUNC(csr2csc, float,  S)
SPARSE_FUNC(csr2csc, double, D)
SPARSE_FUNC(csr2csc, cfloat, C)
SPARSE_FUNC(csr2csc, cdouble,Z)

SPARSE_FUNC_DEF(dense2csr)
SPARSE_FUNC(dense2csr, float,  S)
SPARSE_FUNC(dense2csr, double, D)
SPARSE_FUNC(dense2csr, cfloat, C)
SPARSE_FUNC(dense2csr, cdouble,Z)

SPARSE_FUNC_DEF(dense2csc)
SPARSE_FUNC(dense2csc, float,  S)
SPARSE_FUNC(dense2csc, double, D)
SPARSE_FUNC(dense2csc, cfloat, C)
SPARSE_FUNC(dense2csc, cdouble,Z)

SPARSE_FUNC_DEF(nnz)
SPARSE_FUNC(nnz, float,  S)
SPARSE_FUNC(nnz, double, D)
SPARSE_FUNC(nnz, cfloat, C)
SPARSE_FUNC(nnz, cdouble,Z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

template<typename T, af_sparse_storage storage>
void dense2storage(Array<T> &values, Array<int> &rowIdx, Array<int> &colIdx,
                   const Array<T> in)
{
    const int M = in.dims()[0];
    const int N = in.dims()[1];

    // Create Sparse Matrix Descriptor
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int d = -1;
    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    if(storage == AF_SPARSE_CSR) {
        d = M;
        dir = CUSPARSE_DIRECTION_ROW;
    } else {
        d = N;
        dir = CUSPARSE_DIRECTION_COLUMN;
    }
    Array<int> nnzPerDir = createEmptyArray<int>(dim4(d));

    int nNZ = -1;
    CUSPARSE_CHECK(nnz_func<T>()(
                        getHandle(),
                        dir,
                        M, N,
                        descr,
                        in.get(), in.strides()[1],
                        nnzPerDir.get(), &nNZ));

    if(storage == AF_SPARSE_CSR) {
        rowIdx = createEmptyArray<int>(dim4(M+1));
        colIdx = createEmptyArray<int>(dim4(nNZ));
    } else {
        rowIdx = createEmptyArray<int>(dim4(nNZ));
        colIdx = createEmptyArray<int>(dim4(N+1));
    }
    values = createEmptyArray<T>(dim4(nNZ));

    if(storage == AF_SPARSE_CSR)
        CUSPARSE_CHECK(dense2csr_func<T>()(
                        getHandle(),
                        M, N,
                        descr,
                        in.get(), in.strides()[1],
                        nnzPerDir.get(),
                        values.get(), rowIdx.get(), colIdx.get()));
    else
        CUSPARSE_CHECK(dense2csc_func<T>()(
                        getHandle(),
                        M, N,
                        descr,
                        in.get(), in.strides()[1],
                        nnzPerDir.get(),
                        values.get(), rowIdx.get(), colIdx.get()));

    // Destory Sparse Matrix Descriptor
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
}

#define INSTANTIATE_SPARSE(T)                                                           \
    template void dense2storage<T, AF_SPARSE_CSR>(                                      \
        Array<T> &values, Array<int> &rowIdx, Array<int> &colIdx,                       \
        const Array<T> in);                                                             \
    template void dense2storage<T, AF_SPARSE_CSC>(                                      \
        Array<T> &values, Array<int> &rowIdx, Array<int> &colIdx,                       \
        const Array<T> in);                                                             \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}
