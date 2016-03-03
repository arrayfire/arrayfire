/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/sparse.h>
#include <af/array.h>
#include <sparse_t.hpp>
#include <sparse_matmul.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <err_common.hpp>

using namespace detail;
using af::dim4;

template<typename T>
static inline af_array matmul(const af_sparse_t lhs, const af_array rhs,
                              af_mat_prop optLhs, af_mat_prop optRhs)
{
    return getHandle(detail::matmul<T>(
                lhs.nRows, lhs.nCols, lhs.nNZ,
                getArray<T>(lhs.values), getArray<int>(lhs.rowIdx), getArray<int>(lhs.colIdx),
                getArray<T>(rhs), optLhs, optRhs));
}

af_err af_sparse_matmul(af_array *out,
                        const af_sparse_array lhs_, const af_sparse_array rhs,
                        const af_mat_prop optLhs, const af_mat_prop optRhs)
{
    try {
        af_sparse_t lhs = getSparse(lhs_);

        ArrayInfo lhsInfo = getInfo(lhs.values);
        ArrayInfo rhsInfo = getInfo(rhs);

        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        ARG_ASSERT(1, lhs.storage == AF_SPARSE_CSR);

        if (!(optLhs == AF_MAT_NONE ||
              optLhs == AF_MAT_TRANS ||
              optLhs == AF_MAT_CTRANS)) {
            AF_ERROR("Using this property is not yet supported in sparse matmul", AF_ERR_NOT_SUPPORTED);
        }
        if (optRhs != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in matmul", AF_ERR_NOT_SUPPORTED);
        }

        if (rhsInfo.ndims() > 2) {
            AF_ERROR("Sparse matmul can not be used in batch mode", AF_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        dim4 ldims(lhs.nRows, lhs.nCols);
        int lColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
        int rRowDim = (optRhs == AF_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, ldims[lColDim] == rhsInfo.dims()[rRowDim]);

        af_array output = 0;
        switch(lhs_type) {
            case f32: output = matmul<float  >(lhs, rhs, optLhs, optRhs);   break;
            case c32: output = matmul<cfloat >(lhs, rhs, optLhs, optRhs);   break;
            case f64: output = matmul<double >(lhs, rhs, optLhs, optRhs);   break;
            case c64: output = matmul<cdouble>(lhs, rhs, optLhs, optRhs);   break;
            default:  TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);

    } CATCHALL;
    return AF_SUCCESS;
}
