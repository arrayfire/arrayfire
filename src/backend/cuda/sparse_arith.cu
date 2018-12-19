/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse.hpp>
#include <kernel/sparse_arith.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <cast.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <common/err_common.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <where.hpp>

namespace cuda
{

using namespace common;
using namespace std;

template<typename T>
T getInf()
{
    return scalar<T>(std::numeric_limits<T>::infinity());
}

template<>
cfloat getInf()
{
    return scalar<cfloat, float>(NAN, NAN); // Matches behavior of complex division by 0 in CUDA
}

template<>
cdouble getInf()
{
    return scalar<cdouble, double>(NAN, NAN); // Matches behavior of complex division by 0 in CUDA
}

template<typename T, af_op_t op>
Array<T> arithOpD(const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse)
{
    lhs.eval();
    rhs.eval();

    Array<T> out = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch(op) {
        case af_add_t: out = copyArray<T>(rhs); break;
        case af_sub_t: out = reverse ? copyArray<T>(rhs) : arithOp<T, af_sub_t>(zero, rhs, rhs.dims()); break;
        default      : out = copyArray<T>(rhs);
    }
    out.eval();
    switch(lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                                            rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO", AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse)
{
    lhs.eval();
    rhs.eval();

    SparseArray<T> out = createArrayDataSparseArray<T>(lhs.dims(), lhs.getValues(),
                                                       lhs.getRowIdx(), lhs.getColIdx(),
                                                       lhs.getStorage(), true);
    out.eval();
    switch(lhs.getStorage()) {
        case AF_STORAGE_CSR:
            kernel::sparseArithOpCSR<T, op>(out.getValues(), out.getRowIdx(), out.getColIdx(),
                                            rhs, reverse);
            break;
        case AF_STORAGE_COO:
            kernel::sparseArithOpCOO<T, op>(out.getValues(), out.getRowIdx(), out.getColIdx(),
                                            rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO", AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T>
using csrgeam_def = cusparseStatus_t (*)(cusparseHandle_t, int, int,
        const T*, const cusparseMatDescr_t, int, const T*, const int*, const int*,
        const T*, const cusparseMatDescr_t, int, const T*, const int*, const int*,
        const cusparseMatDescr_t, T*, int*, int*);

#define SPARSE_ARITH_OP_FUNC_DEF( FUNC )            \
template<typename T> FUNC##_def<T> FUNC##_func();

SPARSE_ARITH_OP_FUNC_DEF( csrgeam );

#define SPARSE_ARITH_OP_FUNC( FUNC, TYPE, INFIX )   \
template<> FUNC##_def<TYPE> FUNC##_func<TYPE>()     \
{ return cusparse##INFIX##FUNC; }

SPARSE_ARITH_OP_FUNC(csrgeam, float  , S);
SPARSE_ARITH_OP_FUNC(csrgeam, double , D);
SPARSE_ARITH_OP_FUNC(csrgeam, cfloat , C);
SPARSE_ARITH_OP_FUNC(csrgeam, cdouble, Z);

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs)
{
    lhs.eval();
    rhs.eval();
    af::storage sfmt = lhs.getStorage();

    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);

    const dim4 ldims = lhs.dims();

    const int M = ldims[0];
    const int N = ldims[1];

    const dim_t nnzA = lhs.getNNZ();
    const dim_t nnzB = rhs.getNNZ();

    const int* csrRowPtrA = lhs.getRowIdx().get();
    const int* csrColPtrA = lhs.getColIdx().get();
    const int* csrRowPtrB = rhs.getRowIdx().get();
    const int* csrColPtrB = rhs.getColIdx().get();

    auto outRowIdx = createEmptyArray<int>(dim4(M+1));

    int* csrRowPtrC = outRowIdx.get();
    int baseC, nnzC;
    int* nnzcDevHostPtr = &nnzC;

    cusparseXcsrgeamNnz(sparseHandle(), M, N,
                        desc, nnzA, csrRowPtrA, csrColPtrA,
                        desc, nnzB, csrRowPtrB, csrColPtrB,
                        desc, csrRowPtrC, nnzcDevHostPtr);
    if (NULL != nnzcDevHostPtr) {
        nnzC = *nnzcDevHostPtr;
    } else {
        cudaMemcpyAsync(&nnzC, csrRowPtrC+M, sizeof(int),
                        cudaMemcpyDeviceToHost, cuda::getActiveStream());
        cudaMemcpyAsync(&baseC, csrRowPtrC, sizeof(int),
                        cudaMemcpyDeviceToHost, cuda::getActiveStream());
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        nnzC -= baseC;
    }

    auto outColIdx = createEmptyArray<int>(dim4(nnzC));
    auto outValues = createEmptyArray<T>(dim4(nnzC));

    T alpha = scalar<T>(1);
    T beta  = op == af_sub_t ? scalar<T>(-1) : alpha;

    csrgeam_func<T>()(sparseHandle(), M, N,
                      &alpha, desc, nnzA,
                      lhs.getValues().get(), csrRowPtrA, csrColPtrA,
                      &beta,  desc, nnzB,
                      rhs.getValues().get(), csrRowPtrB, csrColPtrB,
                      desc, outValues.get(), csrRowPtrC, outColIdx.get());

    SparseArray<T> retVal = createArrayDataSparseArray(ldims,
                                 outValues, outRowIdx, outColIdx,
                                 sfmt);
    return retVal;
}

#define INSTANTIATE(T)                                                                              \
    template Array<T> arithOpD<T, af_add_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template Array<T> arithOpD<T, af_sub_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template Array<T> arithOpD<T, af_mul_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template Array<T> arithOpD<T, af_div_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template SparseArray<T> arithOp<T, af_add_t>(const SparseArray<T> &lhs, const Array<T> &rhs,    \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOp<T, af_sub_t>(const SparseArray<T> &lhs, const Array<T> &rhs,    \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOp<T, af_mul_t>(const SparseArray<T> &lhs, const Array<T> &rhs,    \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOp<T, af_div_t>(const SparseArray<T> &lhs, const Array<T> &rhs,    \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOp<T, af_add_t>(const common::SparseArray<T> &lhs,                 \
                                                 const common::SparseArray<T> &rhs);                \
    template SparseArray<T> arithOp<T, af_sub_t>(const common::SparseArray<T> &lhs,                 \
                                                 const common::SparseArray<T> &rhs);                \
    template SparseArray<T> arithOp<T, af_mul_t>(const common::SparseArray<T> &lhs,                 \
                                                 const common::SparseArray<T> &rhs);                \
    template SparseArray<T> arithOp<T, af_div_t>(const common::SparseArray<T> &lhs,                 \
                                                 const common::SparseArray<T> &rhs);

INSTANTIATE(float  )
INSTANTIATE(double )
INSTANTIATE(cfloat )
INSTANTIATE(cdouble)

}

