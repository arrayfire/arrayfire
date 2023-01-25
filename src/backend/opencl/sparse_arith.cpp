/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sparse_arith.hpp>
#include <sparse.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <scan.hpp>
#include <where.hpp>

namespace arrayfire {
namespace opencl {

using namespace common;
using std::numeric_limits;

template<typename T>
T getInf() {
    return scalar<T>(numeric_limits<T>::infinity());
}

template<>
cfloat getInf() {
    return scalar<cfloat, float>(
        NAN, NAN);  // Matches behavior of complex division by 0 in OpenCL
}

template<>
cdouble getInf() {
    return scalar<cdouble, double>(
        NAN, NAN);  // Matches behavior of complex division by 0 in OpenCL
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

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs) {
    lhs.eval();
    rhs.eval();
    af::storage sfmt = lhs.getStorage();

    const dim4 &ldims = lhs.dims();

    const uint M = ldims[0];
    const uint N = ldims[1];

    const dim_t nnzA = lhs.getNNZ();
    const dim_t nnzB = rhs.getNNZ();

    auto temp = createValueArray<int>(dim4(M + 1), scalar<int>(0));
    temp.eval();

    unsigned nnzC = 0;
    kernel::csrCalcOutNNZ(temp, nnzC, M, N, nnzA, lhs.getRowIdx(),
                          lhs.getColIdx(), nnzB, rhs.getRowIdx(),
                          rhs.getColIdx());

    auto outRowIdx = scan<af_add_t, int, int>(temp, 0);

    auto outColIdx = createEmptyArray<int>(dim4(nnzC));
    auto outValues = createEmptyArray<T>(dim4(nnzC));

    kernel::ssArithCSR<T, op>(outValues, outColIdx, outRowIdx, M, N, nnzA,
                              lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                              nnzB, rhs.getValues(), rhs.getRowIdx(),
                              rhs.getColIdx());

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
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace opencl
}  // namespace arrayfire
