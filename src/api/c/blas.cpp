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
                    af_mat_prop optLhs, af_mat_prop optRhs)
{
    return getHandle(detail::matmul<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline af_array dot(const af_array lhs, const af_array rhs,
                    af_mat_prop optLhs, af_mat_prop optRhs)
{
    return getHandle(detail::dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

af_err af_matmul(   af_array *out,
                    const af_array lhs, const af_array rhs,
                    const af_mat_prop optLhs, const af_mat_prop optRhs)
{
    using namespace detail;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        if (!(optLhs == AF_MAT_NONE ||
              optLhs == AF_MAT_TRANS ||
              optLhs == AF_MAT_CTRANS)) {
            AF_ERROR("Using this property is not yet supported in matmul", AF_ERR_NOT_SUPPORTED);
        }

        if (!(optRhs == AF_MAT_NONE ||
              optRhs == AF_MAT_TRANS ||
              optRhs == AF_MAT_CTRANS)) {
            AF_ERROR("Using this property is not yet supported in matmul", AF_ERR_NOT_SUPPORTED);
        }


        if (lhsInfo.ndims() > 2 ||
            rhsInfo.ndims() > 2) {
            AF_ERROR("matmul can not be used in batch mode", AF_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);
        af_array output = 0;

        int aColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
        int bRowDim = (optRhs == AF_MAT_NONE) ? 0 : 1;

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
                    const af_mat_prop optLhs, const af_mat_prop optRhs)
{
    using namespace detail;

    try {
        ArrayInfo lhsInfo = getInfo(lhs);
        ArrayInfo rhsInfo = getInfo(rhs);

        if (optLhs != AF_MAT_NONE && optLhs != AF_MAT_CONJ) {
            AF_ERROR("Using this property is not yet supported in dot", AF_ERR_NOT_SUPPORTED);
        }

        if (optRhs != AF_MAT_NONE && optRhs != AF_MAT_CONJ) {
            AF_ERROR("Using this property is not yet supported in dot", AF_ERR_NOT_SUPPORTED);
        }

        DIM_ASSERT(1, lhsInfo.dims()[0] == rhsInfo.dims()[0]);
        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        if (lhsInfo.ndims() > 1 ||
            rhsInfo.ndims() > 1) {
            AF_ERROR("dot can not be used in batch mode", AF_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        af_array output = 0;

        switch(lhs_type) {
        case f32: output = dot<float  >(lhs, rhs, optLhs, optRhs);    break;
        case c32: output = dot<cfloat >(lhs, rhs, optLhs, optRhs);    break;
        case f64: output = dot<double >(lhs, rhs, optLhs, optRhs);    break;
        case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs);    break;
        default:  TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL
        return AF_SUCCESS;
}
