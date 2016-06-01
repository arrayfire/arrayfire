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
#include <kernel/sparse.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <cast.hpp>
#include <complex.hpp>
#include <err_common.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <where.hpp>

namespace cuda
{

using cusparse::getHandle;
using namespace common;
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

//cusparseStatus_t cusparseZcsr2dense(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *csrValA,
//                                    const int *csrRowPtrA,
//                                    const int *csrColIndA,
//                                    cuDoubleComplex *A, int lda)
template<typename T>
struct csr2dense_func_def_t
{
    typedef cusparseStatus_t (*csr2dense_func_def)( cusparseHandle_t,
                                                    int, int,
                                                    const cusparseMatDescr_t,
                                                    const T *,
                                                    const int *,
                                                    const int *,
                                                    T *, int);
};

//cusparseStatus_t cusparseZcsc2dense(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *cscValA,
//                                    const int *cscRowIndA,
//                                    const int *cscColPtrA,
//                                    cuDoubleComplex *A, int lda)
template<typename T>
struct csc2dense_func_def_t
{
    typedef cusparseStatus_t (*csc2dense_func_def)( cusparseHandle_t,
                                                    int, int,
                                                    const cusparseMatDescr_t,
                                                    const T *,
                                                    const int *,
                                                    const int *,
                                                    T *, int);
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

SPARSE_FUNC_DEF(csr2dense)
SPARSE_FUNC(csr2dense, float,  S)
SPARSE_FUNC(csr2dense, double, D)
SPARSE_FUNC(csr2dense, cfloat, C)
SPARSE_FUNC(csr2dense, cdouble,Z)

SPARSE_FUNC_DEF(csc2dense)
SPARSE_FUNC(csc2dense, float,  S)
SPARSE_FUNC(csc2dense, double, D)
SPARSE_FUNC(csc2dense, cfloat, C)
SPARSE_FUNC(csc2dense, cdouble,Z)

SPARSE_FUNC_DEF(nnz)
SPARSE_FUNC(nnz, float,  S)
SPARSE_FUNC(nnz, double, D)
SPARSE_FUNC(nnz, cfloat, C)
SPARSE_FUNC(nnz, cdouble,Z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in)
{
    Array<uint> nonZeroIdx_ = where<T>(in);
    Array<int> nonZeroIdx = cast<int, uint>(nonZeroIdx_);

    dim_t nNZ = nonZeroIdx.elements();

    Array<int> constNNZ = createValueArray<int>(dim4(nNZ), nNZ);

    Array<int> rowIdx = arithOp<int, af_mod_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
    Array<int> colIdx = arithOp<int, af_div_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
    Array<T>   values = lookup<T, int>(in, nonZeroIdx, 0);

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx, AF_SPARSE_COO);
}

template<typename T, af_sparse_storage storage>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in)
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

    Array<int> rowIdx = createEmptyArray<int>(dim4());
    Array<int> colIdx = createEmptyArray<int>(dim4());

    if(storage == AF_SPARSE_CSR) {
        rowIdx = createEmptyArray<int>(dim4(M+1));
        colIdx = createEmptyArray<int>(dim4(nNZ));
    } else {
        rowIdx = createEmptyArray<int>(dim4(nNZ));
        colIdx = createEmptyArray<int>(dim4(N+1));
    }
    Array<T> values = createEmptyArray<T>(dim4(nNZ));

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

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx, storage);
}


// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in)
{
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));

    const Array<T>   values = in.getValues();
    const Array<int> rowIdx = in.getRowIdx();
    const Array<int> colIdx = in.getColIdx();

    kernel::coo2dense<T>(dense, values, rowIdx, colIdx);

    return dense;
}

template<typename T, af_sparse_storage storage>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in)
{
    // Create Sparse Matrix Descriptor
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int M = in.dims()[0];
    int N = in.dims()[1];
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));
    int d_strides1 = dense.strides()[1];

    if(storage == AF_SPARSE_CSR)
        CUSPARSE_CHECK(csr2dense_func<T>()(
                        getHandle(),
                        M, N,
                        descr,
                        in.getValues().get(),
                        in.getRowIdx().get(),
                        in.getColIdx().get(),
                        dense.get(), d_strides1));
    else
        CUSPARSE_CHECK(csc2dense_func<T>()(
                        getHandle(),
                        M, N,
                        descr,
                        in.getValues().get(),
                        in.getRowIdx().get(),
                        in.getColIdx().get(),
                        dense.get(), d_strides1));

    // Destory Sparse Matrix Descriptor
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));

    return dense;
}

template<typename T, af_sparse_storage src, af_sparse_storage dest>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in)
{
    // Dummy function
    // TODO finish this function when support is required
    SparseArray<T> dense = createEmptySparseArray<T>(in.dims(), in.getNNZ(), dest);

    return dense;
}


#define INSTANTIATE_TO_STORAGE(T, S)                                                                        \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_SPARSE_CSR>(const SparseArray<T> &in);   \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_SPARSE_CSC>(const SparseArray<T> &in);   \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_SPARSE_COO>(const SparseArray<T> &in);   \

#define INSTANTIATE_COO_SPECIAL(T)                                                                      \
    template<> SparseArray<T> sparseConvertDenseToStorage<T, AF_SPARSE_COO>(const Array<T> &in)         \
    { return sparseConvertDenseToCOO<T>(in); }                                                          \
    template<> Array<T> sparseConvertStorageToDense<T, AF_SPARSE_COO>(const SparseArray<T> &in)         \
    { return sparseConvertCOOToDense<T>(in); }                                                          \

#define INSTANTIATE_SPARSE(T)                                                                           \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_SPARSE_CSR>(const Array<T> &in);          \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_SPARSE_CSC>(const Array<T> &in);          \
                                                                                                        \
    template Array<T> sparseConvertStorageToDense<T, AF_SPARSE_CSR>(const SparseArray<T> &in);          \
    template Array<T> sparseConvertStorageToDense<T, AF_SPARSE_CSC>(const SparseArray<T> &in);          \
                                                                                                        \
    INSTANTIATE_COO_SPECIAL(T)                                                                          \
                                                                                                        \
    INSTANTIATE_TO_STORAGE(T, AF_SPARSE_CSR)                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_SPARSE_CSC)                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_SPARSE_COO)                                                            \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}
