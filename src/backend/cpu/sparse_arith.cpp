/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_arith.hpp>
#include <SparseArray.hpp>
#include <optypes.hpp>
#include <sparse.hpp>

#include <kernel/sparse_arith.hpp>

#include <stdexcept>
#include <string>

#include <af/dim4.hpp>
#include <arith.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <err_common.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

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
Array<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse)
{
    lhs.eval();
    rhs.eval();

    Array<T> out = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch(op) {
        case af_add_t: out = copyArray<T>(rhs); break;
        case af_sub_t: out = reverse ? copyArray<T>(rhs) : arithOp<T, af_sub_t>(zero, rhs, rhs.dims()); break;
        case af_mul_t: out = zero; break;
        case af_div_t: out = reverse ? createValueArray(rhs.dims(), getInf<T>()) : zero; break;
        default      : out = copyArray<T>(rhs);
    }
    out.eval();
    switch(lhs.getStorage()) {
        case AF_STORAGE_CSR:
            getQueue().enqueue(kernel::sparseArithOp<T, op, AF_STORAGE_CSR>,
                               out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                               rhs, reverse);
            break;
        case AF_STORAGE_COO:
            getQueue().enqueue(kernel::sparseArithOp<T, op, AF_STORAGE_COO>,
                               out, lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
                               rhs, reverse);
            break;
        default:
            AF_ERROR("Sparse Arithmetic only supported for CSR or COO", AF_ERR_NOT_SUPPORTED);
    }

    return out;
}

#define INSTANTIATE(T)                                                                          \
    template Array<T> arithOp<T, af_add_t>(const SparseArray<T> &lhs, const Array<T> &rhs,      \
                                           const bool reverse);                                 \
    template Array<T> arithOp<T, af_sub_t>(const SparseArray<T> &lhs, const Array<T> &rhs,      \
                                           const bool reverse);                                 \
    template Array<T> arithOp<T, af_mul_t>(const SparseArray<T> &lhs, const Array<T> &rhs,      \
                                           const bool reverse);                                 \
    template Array<T> arithOp<T, af_div_t>(const SparseArray<T> &lhs, const Array<T> &rhs,      \
                                           const bool reverse);                                 \

INSTANTIATE(float  )
INSTANTIATE(double )
INSTANTIATE(cfloat )
INSTANTIATE(cdouble)

}
