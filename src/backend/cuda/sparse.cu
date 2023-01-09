/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cudaDataType.hpp>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <handle.hpp>
#include <kernel/sparse.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <where.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

using namespace common;

// cusparseStatus_t cusparseZdense2csr(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *A, int lda,
//                                    const int *nnzPerRow,
//                                    cuDoubleComplex *csrValA,
//                                    int *csrRowPtrA, int *csrColIndA)
template<typename T>
struct dense2csr_func_def_t {
    typedef cusparseStatus_t (*dense2csr_func_def)(cusparseHandle_t, int, int,
                                                   const cusparseMatDescr_t,
                                                   const T *, int, const int *,
                                                   T *, int *, int *);
};

// cusparseStatus_t cusparseZdense2csc(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *A, int lda,
//                                    const int *nnzPerCol,
//                                    cuDoubleComplex *cscValA,
//                                    int *cscRowIndA, int *cscColPtrA)
template<typename T>
struct dense2csc_func_def_t {
    typedef cusparseStatus_t (*dense2csc_func_def)(cusparseHandle_t, int, int,
                                                   const cusparseMatDescr_t,
                                                   const T *, int, const int *,
                                                   T *, int *, int *);
};

// cusparseStatus_t cusparseZcsr2dense(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *csrValA,
//                                    const int *csrRowPtrA,
//                                    const int *csrColIndA,
//                                    cuDoubleComplex *A, int lda)
template<typename T>
struct csr2dense_func_def_t {
    typedef cusparseStatus_t (*csr2dense_func_def)(cusparseHandle_t, int, int,
                                                   const cusparseMatDescr_t,
                                                   const T *, const int *,
                                                   const int *, T *, int);
};

// cusparseStatus_t cusparseZcsc2dense(cusparseHandle_t handle,
//                                    int m, int n,
//                                    const cusparseMatDescr_t descrA,
//                                    const cuDoubleComplex *cscValA,
//                                    const int *cscRowIndA,
//                                    const int *cscColPtrA,
//                                    cuDoubleComplex *A, int lda)
template<typename T>
struct csc2dense_func_def_t {
    typedef cusparseStatus_t (*csc2dense_func_def)(cusparseHandle_t, int, int,
                                                   const cusparseMatDescr_t,
                                                   const T *, const int *,
                                                   const int *, T *, int);
};

// cusparseStatus_t cusparseZnnz(cusparseHandle_t handle,
//                              cusparseDirection_t dirA,
//                              int m, int n,
//                              const cusparseMatDescr_t descrA,
//                              const cuDoubleComplex *A, int lda,
//                              int *nnzPerRowColumn,
//                              int *nnzTotalDevHostPtr)
template<typename T>
struct nnz_func_def_t {
    typedef cusparseStatus_t (*nnz_func_def)(cusparseHandle_t,
                                             cusparseDirection_t, int, int,
                                             const cusparseMatDescr_t,
                                             const T *, int, int *, int *);
};

// cusparseStatus_t cusparseZgthr(cusparseHandle_t handle,
//                               int nnz,
//                               const cuDoubleComplex *y,
//                               cuDoubleComplex *xVal, const int *xInd,
//                               cusparseIndexBase_t idxBase)
template<typename T>
struct gthr_func_def_t {
    typedef cusparseStatus_t (*gthr_func_def)(cusparseHandle_t, int, const T *,
                                              T *, const int *,
                                              cusparseIndexBase_t);
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

/// Newer versions of cusparse use matrix descriptor instead of types encoded in
/// their names
#if CUSPARSE_VERSION < 11300
SPARSE_FUNC_DEF(dense2csr)
SPARSE_FUNC(dense2csr, float, S)
SPARSE_FUNC(dense2csr, double, D)
SPARSE_FUNC(dense2csr, cfloat, C)
SPARSE_FUNC(dense2csr, cdouble, Z)

SPARSE_FUNC_DEF(dense2csc)
SPARSE_FUNC(dense2csc, float, S)
SPARSE_FUNC(dense2csc, double, D)
SPARSE_FUNC(dense2csc, cfloat, C)
SPARSE_FUNC(dense2csc, cdouble, Z)

SPARSE_FUNC_DEF(csr2dense)
SPARSE_FUNC(csr2dense, float, S)
SPARSE_FUNC(csr2dense, double, D)
SPARSE_FUNC(csr2dense, cfloat, C)
SPARSE_FUNC(csr2dense, cdouble, Z)

SPARSE_FUNC_DEF(csc2dense)
SPARSE_FUNC(csc2dense, float, S)
SPARSE_FUNC(csc2dense, double, D)
SPARSE_FUNC(csc2dense, cfloat, C)
SPARSE_FUNC(csc2dense, cdouble, Z)

SPARSE_FUNC_DEF(gthr)
SPARSE_FUNC(gthr, float, S)
SPARSE_FUNC(gthr, double, D)
SPARSE_FUNC(gthr, cfloat, C)
SPARSE_FUNC(gthr, cdouble, Z)
#endif

SPARSE_FUNC_DEF(nnz)
SPARSE_FUNC(nnz, float, S)
SPARSE_FUNC(nnz, double, D)
SPARSE_FUNC(nnz, cfloat, C)
SPARSE_FUNC(nnz, cdouble, Z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in) {
    Array<uint> nonZeroIdx_ = where<T>(in);
    Array<int> nonZeroIdx   = cast<int, uint>(nonZeroIdx_);

    dim_t nNZ = nonZeroIdx.elements();

    Array<int> constDim = createValueArray<int>(dim4(nNZ), in.dims()[0]);

    Array<int> rowIdx =
        arithOp<int, af_mod_t>(nonZeroIdx, constDim, nonZeroIdx.dims());
    Array<int> colIdx =
        arithOp<int, af_div_t>(nonZeroIdx, constDim, nonZeroIdx.dims());

    Array<T> values = copyArray<T>(in);
    values.modDims(dim4(values.elements()));
    values = lookup<T, int>(values, nonZeroIdx, 0);

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx,
                                         AF_STORAGE_COO);
}

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in) {
    const int M = in.dims()[0];
    const int N = in.dims()[1];

#if CUSPARSE_VERSION < 11300
    // Create Sparse Matrix Descriptor
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int d                   = -1;
    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    if (stype == AF_STORAGE_CSR) {
        d   = M;
        dir = CUSPARSE_DIRECTION_ROW;
    } else {
        d   = N;
        dir = CUSPARSE_DIRECTION_COLUMN;
    }
    Array<int> nnzPerDir = createEmptyArray<int>(dim4(d));

    int nNZ = -1;
    CUSPARSE_CHECK(nnz_func<T>()(sparseHandle(), dir, M, N, descr, in.get(),
                                 in.strides()[1], nnzPerDir.get(), &nNZ));

    Array<int> rowIdx = createEmptyArray<int>(dim4());
    Array<int> colIdx = createEmptyArray<int>(dim4());

    if (stype == AF_STORAGE_CSR) {
        rowIdx = createEmptyArray<int>(dim4(M + 1));
        colIdx = createEmptyArray<int>(dim4(nNZ));
    } else {
        rowIdx = createEmptyArray<int>(dim4(nNZ));
        colIdx = createEmptyArray<int>(dim4(N + 1));
    }
    Array<T> values = createEmptyArray<T>(dim4(nNZ));

    if (stype == AF_STORAGE_CSR) {
        CUSPARSE_CHECK(dense2csr_func<T>()(
            sparseHandle(), M, N, descr, in.get(), in.strides()[1],
            nnzPerDir.get(), values.get(), rowIdx.get(), colIdx.get()));
    } else {
        CUSPARSE_CHECK(dense2csc_func<T>()(
            sparseHandle(), M, N, descr, in.get(), in.strides()[1],
            nnzPerDir.get(), values.get(), rowIdx.get(), colIdx.get()));
    }
    // Destory Sparse Matrix Descriptor
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx,
                                         stype);
#else
    auto matA = denMatDescriptor(in);
    cusparseSpMatDescr_t matB;

    Array<int> d_offsets = createEmptyArray<int>(0);

    if (stype == AF_STORAGE_CSR) {
        d_offsets = createEmptyArray<int>(M + 1);
        // Create sparse matrix B in CSR format
        CUSPARSE_CHECK(
            cusparseCreateCsr(&matB, M, N, 0, d_offsets.get(), nullptr, nullptr,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, getType<T>()));
    } else {
        d_offsets = createEmptyArray<int>(N + 1);
        CUSPARSE_CHECK(
            cusparseCreateCsc(&matB, M, N, 0, d_offsets.get(), nullptr, nullptr,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, getType<T>()));
    }

    // allocate an external buffer if needed
    size_t bufferSize;
    CUSPARSE_CHECK(cusparseDenseToSparse_bufferSize(
        sparseHandle(), matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize));

    auto dBuffer = memAlloc<char>(bufferSize);

    // execute Sparse to Dense conversion
    CUSPARSE_CHECK(cusparseDenseToSparse_analysis(
        sparseHandle(), matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer.get()));
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CUSPARSE_CHECK(
        cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz));

    auto d_ind    = createEmptyArray<int>(nnz);
    auto d_values = createEmptyArray<T>(nnz);
    // allocate CSR column indices and values
    // reset offsets, column indices, and values pointers
    if (stype == AF_STORAGE_CSR) {
        // Create sparse matrix B in CSR format
        // reset offsets, column indices, and values pointers
        CUSPARSE_CHECK(cusparseCsrSetPointers(matB, d_offsets.get(),
                                              d_ind.get(), d_values.get()));

    } else {
        // reset offsets, column indices, and values pointers
        CUSPARSE_CHECK(cusparseCscSetPointers(matB, d_offsets.get(),
                                              d_ind.get(), d_values.get()));
    }
    // execute Sparse to Dense conversion
    CUSPARSE_CHECK(cusparseDenseToSparse_convert(
        sparseHandle(), matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer.get()));

    if (stype == AF_STORAGE_CSR) {
        size_t pBufferSizeInBytes = 0;
        auto desc                 = make_handle<cusparseMatDescr_t>();
        CUSPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(
            sparseHandle(), M, N, nnz, d_offsets.get(), d_ind.get(),
            &pBufferSizeInBytes));
        auto pBuffer = memAlloc<char>(pBufferSizeInBytes);
        Array<int> P = createEmptyArray<int>(nnz);
        CUSPARSE_CHECK(
            cusparseCreateIdentityPermutation(sparseHandle(), nnz, P.get()));
        CUSPARSE_CHECK(cusparseXcsrsort(
            sparseHandle(), M, N, nnz, desc, (int *)d_offsets.get(),
            (int *)d_ind.get(), P.get(), pBuffer.get()));
        d_values = lookup(d_values, P, 0);
        return createArrayDataSparseArray<T>(in.dims(), d_values, d_offsets,
                                             d_ind, stype, false);
    } else {
        return createArrayDataSparseArray<T>(in.dims(), d_values, d_ind,
                                             d_offsets, stype, false);
    }
#endif
}

// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in) {
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));

    const Array<T> values   = in.getValues();
    const Array<int> rowIdx = in.getRowIdx();
    const Array<int> colIdx = in.getColIdx();

    kernel::coo2dense<T>(dense, values, rowIdx, colIdx);

    return dense;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in) {
    // Create Sparse Matrix Descriptor
#if CUSPARSE_VERSION < 11300
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int M          = in.dims()[0];
    int N          = in.dims()[1];
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));
    int d_strides1 = dense.strides()[1];

    if (stype == AF_STORAGE_CSR) {
        CUSPARSE_CHECK(
            csr2dense_func<T>()(sparseHandle(), M, N, descr,
                                in.getValues().get(), in.getRowIdx().get(),
                                in.getColIdx().get(), dense.get(), d_strides1));
    } else {
        CUSPARSE_CHECK(
            csc2dense_func<T>()(sparseHandle(), M, N, descr,
                                in.getValues().get(), in.getRowIdx().get(),
                                in.getColIdx().get(), dense.get(), d_strides1));
    }

    // Destory Sparse Matrix Descriptor
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
#else
    unique_handle<cusparseSpMatDescr_t> inhandle = cusparseDescriptor(in);

    Array<T> dense = createEmptyArray<T>(in.dims());
    unique_handle<cusparseDnMatDescr_t> outhandle = denMatDescriptor(dense);

    size_t bufferSize = 0;
    cusparseSparseToDense_bufferSize(sparseHandle(), inhandle, outhandle,
                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                     &bufferSize);

    auto dBuffer = memAlloc<char>(bufferSize);
    cusparseSparseToDense(sparseHandle(), inhandle, outhandle,
                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, dBuffer.get());

#endif

    return dense;
}

template<typename T, af_storage dest, af_storage src>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in) {
    using std::shared_ptr;
    in.eval();

    int nNZ                  = in.getNNZ();
    SparseArray<T> converted = createEmptySparseArray<T>(in.dims(), nNZ, dest);

    if (src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
        // Copy colIdx as is
        CUDA_CHECK(
            cudaMemcpyAsync(converted.getColIdx().get(), in.getColIdx().get(),
                            in.getColIdx().elements() * sizeof(int),
                            cudaMemcpyDeviceToDevice, getActiveStream()));

        // cusparse function to expand compressed row into coordinate
        CUSPARSE_CHECK(cusparseXcsr2coo(
            sparseHandle(), in.getRowIdx().get(), nNZ, in.dims()[0],
            converted.getRowIdx().get(), CUSPARSE_INDEX_BASE_ZERO));

        // Call sort
        size_t pBufferSizeInBytes = 0;
        CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
            sparseHandle(), in.dims()[0], in.dims()[1], nNZ,
            converted.getRowIdx().get(), converted.getColIdx().get(),
            &pBufferSizeInBytes));
        auto pBuffer = memAlloc<char>(pBufferSizeInBytes);

        // shared_ptr<int> P(memAlloc<int>(nNZ).release(), memFree<int>);
        Array<int> P = createEmptyArray<int>(nNZ);
        CUSPARSE_CHECK(
            cusparseCreateIdentityPermutation(sparseHandle(), nNZ, P.get()));

        CUSPARSE_CHECK(cusparseXcoosortByColumn(
            sparseHandle(), in.dims()[0], in.dims()[1], nNZ,
            converted.getRowIdx().get(), converted.getColIdx().get(), P.get(),
            pBuffer.get()));

        converted.getValues() = lookup<T, int>(in.getValues(), P, 0);

    } else if (src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
        // The cusparse csr sort function is not behaving correctly.
        // So the work around is to convert the COO into row major and then
        // convert it to CSR

        int M = in.dims()[0];
        int N = in.dims()[1];
        // Deep copy input into temporary COO Row Major
        SparseArray<T> cooT = createArrayDataSparseArray<T>(
            in.dims(), in.getValues(), in.getRowIdx(), in.getColIdx(),
            in.getStorage(), true);

        // Call sort to convert column major to row major
        {
            size_t pBufferSizeInBytes = 0;
            CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
                sparseHandle(), M, N, nNZ, cooT.getRowIdx().get(),
                cooT.getColIdx().get(), &pBufferSizeInBytes));
            auto pBuffer = memAlloc<char>(pBufferSizeInBytes);

            Array<int> P = createEmptyArray<int>(nNZ);
            CUSPARSE_CHECK(cusparseCreateIdentityPermutation(sparseHandle(),
                                                             nNZ, P.get()));

            CUSPARSE_CHECK(cusparseXcoosortByRow(
                sparseHandle(), M, N, nNZ, cooT.getRowIdx().get(),
                cooT.getColIdx().get(), P.get(), pBuffer.get()));

            converted.getValues() = lookup<T, int>(in.getValues(), P, 0);
        }

        // Copy values and colIdx as is
        copyArray<int, int>(converted.getColIdx(), cooT.getColIdx());

        // cusparse function to compress row from coordinate
        CUSPARSE_CHECK(cusparseXcoo2csr(sparseHandle(), cooT.getRowIdx().get(),
                                        nNZ, M, converted.getRowIdx().get(),
                                        CUSPARSE_INDEX_BASE_ZERO));

        // No need to call CSRSORT

    } else {
        // Should never come here
        AF_ERROR("CUDA Backend invalid conversion combination",
                 AF_ERR_NOT_SUPPORTED);
    }

    return converted;
}

#define INSTANTIATE_TO_STORAGE(T, S)                     \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>( \
        const SparseArray<T> &in);

#define INSTANTIATE_COO_SPECIAL(T)                                 \
    template<>                                                     \
    SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>( \
        const Array<T> &in) {                                      \
        return sparseConvertDenseToCOO<T>(in);                     \
    }                                                              \
    template<>                                                     \
    Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(       \
        const SparseArray<T> &in) {                                \
        return sparseConvertCOOToDense<T>(in);                     \
    }

#define INSTANTIATE_SPARSE(T)                                               \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>( \
        const Array<T> &in);                                                \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>( \
        const Array<T> &in);                                                \
                                                                            \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(       \
        const SparseArray<T> &in);                                          \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(       \
        const SparseArray<T> &in);                                          \
                                                                            \
    INSTANTIATE_COO_SPECIAL(T)                                              \
                                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}  // namespace cuda
}  // namespace arrayfire
