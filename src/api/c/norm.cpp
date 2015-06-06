/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include <af/defines.h>
#include <af/traits.hpp>
#include <af/constants.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <math.hpp>
#include <arith.hpp>
#include <lu.hpp>
#include <reduce.hpp>
#include <complex.hpp>

using af::dim4;
using namespace detail;

template<typename T>
double matrixNorm(const Array<T> &A, double p)
{
    if (p == 1) {
        Array<T> colSum = reduce<af_add_t, T, T>(A, 0);
        return reduce_all<af_max_t, T, T>(colSum);
    } else if (p == af::Inf) {
        Array<T> rowSum = reduce<af_add_t, T, T>(A, 1);
        return reduce_all<af_max_t, T, T>(rowSum);
    }

    AF_ERROR("This type of norm is not supported in ArrayFire\n", AF_ERR_NOT_SUPPORTED);
}

template<typename T>
double vectorNorm(const Array<T> &A, double p)
{
    if (p == 1) {
        return reduce_all<af_add_t, T, T>(A);
    } else if (p == af::Inf) {
        return reduce_all<af_max_t, T, T>(A);
    } else if (p == 2) {
        Array<T> A_sq = arithOp<T, af_mul_t>(A, A, A.dims());
        return std::sqrt(reduce_all<af_add_t, T, T>(A_sq));
    }

    Array<T> P = createValueArray<T>(A.dims(), scalar<T>(p));
    Array<T> A_p = arithOp<T, af_pow_t>(A, P, A.dims());
    return std::pow(reduce_all<af_add_t, T, T>(A_p), T(1.0/p));
}

template<typename T>
double LPQNorm(const Array<T> &A, double p, double q)
{
    Array<T> A_p_norm = createEmptyArray<T>(dim4());

    if (p == 1) {
        A_p_norm = reduce<af_add_t, T, T>(A, 0);
    } else {
        Array<T> P = createValueArray<T>(A.dims(), scalar<T>(p));
        Array<T> invP = createValueArray<T>(A.dims(), scalar<T>(1.0/p));

        Array<T> A_p = arithOp<T, af_pow_t>(A, P, A.dims());
        Array<T> A_p_sum = reduce<af_add_t, T, T>(A_p, 0);
        A_p_norm = arithOp<T, af_pow_t>(A_p_sum, invP, invP.dims());
    }

    if (q == 1) {
        return reduce_all<af_add_t, T, T>(A_p_norm);
    }

    Array<T> Q = createValueArray<T>(A_p_norm.dims(), scalar<T>(q));
    Array<T> A_p_norm_q = arithOp<T, af_pow_t>(A_p_norm, Q, Q.dims());

    return std::pow(reduce_all<af_add_t, T, T>(A_p_norm_q), T(1.0/q));
}

template<typename T>
double norm(const af_array a, const af_norm_type type, const double p, const double q)
{

    typedef typename af::dtype_traits<T>::base_type BT;

    const Array<BT> A = abs<BT, T>(getArray<T>(a));

    switch (type) {

    case AF_NORM_EUCLID:
        return vectorNorm(A, 2);

    case AF_NORM_VECTOR_1:
        return vectorNorm(A, 1);

    case AF_NORM_VECTOR_INF:
        return vectorNorm(A, af::Inf);

    case AF_NORM_VECTOR_P:
        return vectorNorm(A, p);

    case AF_NORM_MATRIX_1:
        return matrixNorm(A, 1);

    case AF_NORM_MATRIX_INF:
        return matrixNorm(A, af::Inf);

    case AF_NORM_MATRIX_2:
        return matrixNorm(A, 2);

    case AF_NORM_MATRIX_L_PQ:
        return LPQNorm(A, p, q);

    default:
        AF_ERROR("This type of norm is not supported in ArrayFire\n", AF_ERR_NOT_SUPPORTED);
    }
}

af_err af_norm(double *out, const af_array in,
               const af_norm_type type, const double p, const double q)
{

    try {
        ArrayInfo i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype i_type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());                       // Only floating and complex types

        *out = 0;

        switch(i_type) {
        case f32: *out = norm<float  >(in, type, p, q);  break;
        case f64: *out = norm<double >(in, type, p, q);  break;
        case c32: *out = norm<cfloat >(in, type, p, q);  break;
        case c64: *out = norm<cdouble>(in, type, p, q);  break;
        default:  TYPE_ERROR(1, i_type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
