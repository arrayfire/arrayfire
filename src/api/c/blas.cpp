/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/blas.h>
#include <blas.hpp>
#include <handle.hpp>
#include <Array.hpp>
#include <af/array.h>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <backend.hpp>

template<typename T>
static inline af_array matmul(const af_array lhs, const af_array rhs,
                    af_transpose_t optLhs, af_transpose_t optRhs)
{
    return getHandle(detail::matmul<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline af_array dot(const af_array lhs, const af_array rhs,
                    af_transpose_t optLhs, af_transpose_t optRhs)
{
    return getHandle(detail::dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

af_err af_matmul(   af_array *out,
                    const af_array lhs, const af_array rhs,
                    af_transpose_t optLhs, af_transpose_t optRhs)
{
    using namespace detail;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        TYPE_ASSERT(lhs_type == rhs_type);
        af_array output = 0;

        int aColDim = (optLhs == AF_NO_TRANS) ? 1 : 0;
        int bRowDim = (optRhs == AF_NO_TRANS) ? 0 : 1;

        DIM_ASSERT(1, lhsInfo.dims()[aColDim] == rhsInfo.dims()[bRowDim]);

        switch(lhs_type) {
        case f32: output = matmul<float  >(lhs, rhs, optLhs, optRhs);   break;
        case c32: output = matmul<cfloat >(lhs, rhs, optLhs, optRhs);   break;
        case f64: output = matmul<double >(lhs, rhs, optLhs, optRhs);   break;
        case c64: output = matmul<cdouble>(lhs, rhs, optLhs, optRhs);   break;
        default:  TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_dot(      af_array *out,
                    const af_array lhs, const af_array rhs,
                    af_transpose_t optLhs, af_transpose_t optRhs)
{
    using namespace detail;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        DIM_ASSERT(1, lhsInfo.dims()[0] == rhsInfo.dims()[0]);
        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        TYPE_ASSERT(lhs_type == rhs_type);

        af_array output = 0;

        switch(lhs_type) {
        case f32: output = dot<float  >(lhs, rhs, optLhs, optRhs);   break;
            //case c32: output = dot<cfloat >(lhs, rhs, optLhs, optRhs);   break;
        case f64: output = dot<double >(lhs, rhs, optLhs, optRhs);   break;
            //case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs);   break;
        default:  TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL
        return AF_SUCCESS;
}
