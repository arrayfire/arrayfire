/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_arith.hpp>
#include <common/SparseArray.hpp>
#include <sparse.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>
#include <arith.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <common/err_common.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <kernel/sparse_arith.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu
{

using namespace common;

template<typename T>
T getInf()
{
    return scalar<T>(std::numeric_limits<T>::infinity());
}

template<>
cfloat getInf()
{
    return scalar<cfloat, float>(
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()
            );
}

template<>
cdouble getInf()
{
    return scalar<cdouble, double>(
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
            );
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
            getQueue().enqueue(kernel::sparseArithOpD<T, op, AF_STORAGE_CSR>,
                               out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                               rhs, reverse);
            break;
        case AF_STORAGE_COO:
            getQueue().enqueue(kernel::sparseArithOpD<T, op, AF_STORAGE_COO>,
                               out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                               rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO", AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOpS(const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse)
{
    lhs.eval();
    rhs.eval();

    SparseArray<T> out = createArrayDataSparseArray<T>(lhs.dims(), lhs.getValues(),
                                                       lhs.getRowIdx(), lhs.getColIdx(),
                                                       lhs.getStorage(), true);
    out.eval();
    switch(out.getStorage()) {
        case AF_STORAGE_CSR:
            getQueue().enqueue(kernel::sparseArithOpS<T, op, AF_STORAGE_CSR>,
                               out.getValues(), out.getRowIdx(), out.getColIdx(),
                               rhs, reverse);
            break;
        case AF_STORAGE_COO:
            getQueue().enqueue(kernel::sparseArithOpS<T, op, AF_STORAGE_COO>,
                               out.getValues(), out.getRowIdx(), out.getColIdx(),
                               rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO", AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs)
{
    af::storage sfmt = lhs.getStorage();

    lhs.eval();
    rhs.eval();

    const dim4 dims = lhs.dims();
    const uint M = dims[0];
    const uint N = dims[1];

    auto rowArr = createEmptyArray<int>(dim4(M+1));

    getQueue().enqueue(kernel::calcOutNNZ, rowArr, M, N,
                       lhs.getRowIdx(), lhs.getColIdx(),
                       rhs.getRowIdx(), rhs.getColIdx());
    getQueue().sync();

    uint nnz = rowArr.get()[M];
    auto out = createEmptySparseArray<T>(dims, nnz, sfmt);
    out.eval();

    copyArray(out.getRowIdx(), rowArr);

    getQueue().enqueue(kernel::sparseArithOp<T, op>,
                       out.getValues(), out.getColIdx(),
                       out.getRowIdx(), M,
                       lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                       rhs.getValues(), rhs.getRowIdx(), rhs.getColIdx());
    return out;
}

#define INSTANTIATE(T)                                                                              \
    template Array<T> arithOpD<T, af_add_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template Array<T> arithOpD<T, af_sub_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                           const bool reverse);                                     \
    template Array<T> arithOpD<T, af_mul_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template Array<T> arithOpD<T, af_div_t>(const SparseArray<T> &lhs, const Array<T> &rhs,         \
                                            const bool reverse);                                    \
    template SparseArray<T> arithOpS<T, af_add_t>(const SparseArray<T> &lhs, const Array<T> &rhs,   \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOpS<T, af_sub_t>(const SparseArray<T> &lhs, const Array<T> &rhs,   \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOpS<T, af_mul_t>(const SparseArray<T> &lhs, const Array<T> &rhs,   \
                                                  const bool reverse);                              \
    template SparseArray<T> arithOpS<T, af_div_t>(const SparseArray<T> &lhs, const Array<T> &rhs,   \
                                                 const bool reverse);                               \
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
