/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sparse_arith.hpp>
#include <sparse_arith.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/unique_handle.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <cusparse.hpp>
#include <cusparse_descriptor_helpers.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <where.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace cuda {

using namespace common;
using std::numeric_limits;

template<typename T>
T getInf() {
    return scalar<T>(numeric_limits<T>::infinity());
}

template<>
cfloat getInf() {
    return scalar<cfloat, float>(
        NAN, NAN);  // Matches behavior of complex division by 0 in CUDA
}

template<>
cdouble getInf() {
    return scalar<cdouble, double>(
        NAN, NAN);  // Matches behavior of complex division by 0 in CUDA
}

template<typename T, af_op_t op>
Array<T> arithOpD(const SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse) {
    lhs.eval();
    rhs.eval();

    Array<T> out  = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch (op) {
        case af_add_t: out = copyArray<T>(rhs); break;
        case af_sub_t:
            out = reverse ? copyArray<T>(rhs)
                          : arithOp<T, af_sub_t>(zero, rhs, rhs.dims());
            break;
        default: out = copyArray<T>(rhs);
    }
    out.eval();
    switch (lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out, lhs.getValues(),
                                            lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out, lhs.getValues(),
                                            lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs,
                       const bool reverse) {
    lhs.eval();
    rhs.eval();

    SparseArray<T> out = createArrayDataSparseArray<T>(
        lhs.dims(), lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
        lhs.getStorage(), true);
    out.eval();
    switch (lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out.getValues(), out.getRowIdx(),
                                            out.getColIdx(), rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out.getValues(), out.getRowIdx(),
                                            out.getColIdx(), rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

#define SPARSE_ARITH_OP_FUNC_DEF(FUNC) \
    template<typename T>               \
    FUNC##_def<T> FUNC##_func();

#define SPARSE_ARITH_OP_FUNC(FUNC, TYPE, INFIX) \
    template<>                                  \
    FUNC##_def<TYPE> FUNC##_func<TYPE>() {      \
        return cusparse##INFIX##FUNC;           \
    }

#if CUSPARSE_VERSION >= 11000

template<typename T>
using csrgeam2_bufferSizeExt_def = cusparseStatus_t (*)(
    cusparseHandle_t, int, int, const T *, const cusparseMatDescr_t, int,
    const T *, const int *, const int *, const T *, const cusparseMatDescr_t,
    int, const T *, const int *, const int *, const cusparseMatDescr_t,
    const T *, const int *, const int *, size_t *);

#define SPARSE_ARITH_OP_BUFFER_SIZE_FUNC_DEF(FUNC) \
    template<typename T>                           \
    FUNC##_def<T> FUNC##_func();

SPARSE_ARITH_OP_BUFFER_SIZE_FUNC_DEF(csrgeam2_bufferSizeExt);

#define SPARSE_ARITH_OP_BUFFER_SIZE_FUNC(FUNC, TYPE, INFIX) \
    template<>                                              \
    FUNC##_def<TYPE> FUNC##_func<TYPE>() {                  \
        return cusparse##INFIX##FUNC;                       \
    }

SPARSE_ARITH_OP_BUFFER_SIZE_FUNC(csrgeam2_bufferSizeExt, float, S);
SPARSE_ARITH_OP_BUFFER_SIZE_FUNC(csrgeam2_bufferSizeExt, double, D);
SPARSE_ARITH_OP_BUFFER_SIZE_FUNC(csrgeam2_bufferSizeExt, cfloat, C);
SPARSE_ARITH_OP_BUFFER_SIZE_FUNC(csrgeam2_bufferSizeExt, cdouble, Z);

template<typename T>
using csrgeam2_def = cusparseStatus_t (*)(cusparseHandle_t, int, int, const T *,
                                          const cusparseMatDescr_t, int,
                                          const T *, const int *, const int *,
                                          const T *, const cusparseMatDescr_t,
                                          int, const T *, const int *,
                                          const int *, const cusparseMatDescr_t,
                                          T *, int *, int *, void *);

SPARSE_ARITH_OP_FUNC_DEF(csrgeam2);

SPARSE_ARITH_OP_FUNC(csrgeam2, float, S);
SPARSE_ARITH_OP_FUNC(csrgeam2, double, D);
SPARSE_ARITH_OP_FUNC(csrgeam2, cfloat, C);
SPARSE_ARITH_OP_FUNC(csrgeam2, cdouble, Z);

#else

template<typename T>
using csrgeam_def = cusparseStatus_t (*)(cusparseHandle_t, int, int, const T *,
                                         const cusparseMatDescr_t, int,
                                         const T *, const int *, const int *,
                                         const T *, const cusparseMatDescr_t,
                                         int, const T *, const int *,
                                         const int *, const cusparseMatDescr_t,
                                         T *, int *, int *);

SPARSE_ARITH_OP_FUNC_DEF(csrgeam);

SPARSE_ARITH_OP_FUNC(csrgeam, float, S);
SPARSE_ARITH_OP_FUNC(csrgeam, double, D);
SPARSE_ARITH_OP_FUNC(csrgeam, cfloat, C);
SPARSE_ARITH_OP_FUNC(csrgeam, cdouble, Z);

#endif

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs) {
    af::storage sfmt = lhs.getStorage();
    auto ldesc       = make_handle<cusparseMatDescr_t>();
    auto rdesc       = make_handle<cusparseMatDescr_t>();
    auto odesc       = make_handle<cusparseMatDescr_t>();

    const dim4 ldims      = lhs.dims();
    const int M           = ldims[0];
    const int N           = ldims[1];
    const dim_t nnzA      = lhs.getNNZ();
    const dim_t nnzB      = rhs.getNNZ();
    const int *csrRowPtrA = lhs.getRowIdx().get();
    const int *csrColPtrA = lhs.getColIdx().get();
    const int *csrRowPtrB = rhs.getRowIdx().get();
    const int *csrColPtrB = rhs.getColIdx().get();

    int baseC, nnzC = M + 1;

    auto nnzDevHostPtr = memAlloc<int>(1);
    auto outRowIdx     = createValueArray<int>(M + 1, 0);

    T alpha = scalar<T>(1);
    T beta  = op == af_sub_t ? scalar<T>(-1) : scalar<T>(1);

    T *csrValC      = nullptr;
    int *csrColIndC = nullptr;

#if CUSPARSE_VERSION < 11000
    CUSPARSE_CHECK(cusparseXcsrgeamNnz(
        sparseHandle(), M, N, ldesc, nnzA, csrRowPtrA, csrColPtrA, rdesc, nnzB,
        csrRowPtrB, csrColPtrB, odesc, outRowIdx.get(), nnzDevHostPtr.get()));
#else
    size_t pBufferSize = 0;

    CUSPARSE_CHECK(csrgeam2_bufferSizeExt_func<T>()(
        sparseHandle(), M, N, &alpha, ldesc, nnzA, lhs.getValues().get(),
        csrRowPtrA, csrColPtrA, &beta, rdesc, nnzB, rhs.getValues().get(),
        csrRowPtrB, csrColPtrB, odesc, csrValC, outRowIdx.get(), csrColIndC,
        &pBufferSize));

    auto tmpBuffer = memAlloc<char>(pBufferSize);
    CUSPARSE_CHECK(cusparseXcsrgeam2Nnz(
        sparseHandle(), M, N, ldesc, nnzA, csrRowPtrA, csrColPtrA, rdesc, nnzB,
        csrRowPtrB, csrColPtrB, odesc, outRowIdx.get(), nnzDevHostPtr.get(),
        tmpBuffer.get()));
#endif
    if (NULL != nnzDevHostPtr) {
        CUDA_CHECK(cudaMemcpyAsync(&nnzC, nnzDevHostPtr.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(&nnzC, outRowIdx.get() + M, sizeof(int),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(&baseC, outRowIdx.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        nnzC -= baseC;
    }
    auto outColIdx = createEmptyArray<int>(nnzC);
    auto outValues = createEmptyArray<T>(nnzC);

#if CUSPARSE_VERSION < 11000
    CUSPARSE_CHECK(csrgeam_func<T>()(
        sparseHandle(), M, N, &alpha, ldesc, nnzA, lhs.getValues().get(),
        csrRowPtrA, csrColPtrA, &beta, rdesc, nnzB, rhs.getValues().get(),
        csrRowPtrB, csrColPtrB, odesc, outValues.get(), outRowIdx.get(),
        outColIdx.get()));
#else
    CUSPARSE_CHECK(csrgeam2_func<T>()(
        sparseHandle(), M, N, &alpha, ldesc, nnzA, lhs.getValues().get(),
        csrRowPtrA, csrColPtrA, &beta, rdesc, nnzB, rhs.getValues().get(),
        csrRowPtrB, csrColPtrB, odesc, outValues.get(), outRowIdx.get(),
        outColIdx.get(), tmpBuffer.get()));
#endif
    SparseArray<T> retVal = createArrayDataSparseArray(
        ldims, outValues, outRowIdx, outColIdx, sfmt);
    return retVal;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> arithOpD<T, af_add_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_sub_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_mul_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, af_div_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace cuda
}  // namespace arrayfire
