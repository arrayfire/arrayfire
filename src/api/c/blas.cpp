/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>
#include <blas.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <sparse_blas.hpp>
#include <sparse_handle.hpp>
#include <af/array.h>
#include <af/blas.h>
#include <af/defines.h>

template<typename T>
static inline af_array sparseMatmul(const af_array lhs, const af_array rhs,
                                    af_mat_prop optLhs, af_mat_prop optRhs) {
    return getHandle(detail::matmul<T>(getSparseArray<T>(lhs), getArray<T>(rhs),
                                       optLhs, optRhs));
}

template<typename T>
static inline af_array matmul(const af_array lhs, const af_array rhs,
                              af_mat_prop optLhs, af_mat_prop optRhs) {
    return getHandle(
        detail::matmul<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline af_array dot(const af_array lhs, const af_array rhs,
                           af_mat_prop optLhs, af_mat_prop optRhs) {
    return getHandle(
        detail::dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

af_err af_sparse_matmul(af_array *out, const af_array lhs, const af_array rhs,
                        const af_mat_prop optLhs, const af_mat_prop optRhs) {
    using namespace detail;

    try {
        common::SparseArrayBase lhsBase = getSparseArrayBase(lhs);
        const ArrayInfo &rhsInfo        = getInfo(rhs);

        ARG_ASSERT(2,
                   lhsBase.isSparse() == true && rhsInfo.isSparse() == false);

        af_dtype lhs_type = lhsBase.getType();
        af_dtype rhs_type = rhsInfo.getType();

        ARG_ASSERT(1, lhsBase.getStorage() == AF_STORAGE_CSR);

        if (!(optLhs == AF_MAT_NONE || optLhs == AF_MAT_TRANS ||
              optLhs == AF_MAT_CTRANS)) {  // Note the ! operator.
            AF_ERROR(
                "Using this property is not yet supported in sparse matmul",
                AF_ERR_NOT_SUPPORTED);
        }

        // No transpose options for RHS
        if (optRhs != AF_MAT_NONE) {
            AF_ERROR("Using this property is not yet supported in matmul",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (rhsInfo.ndims() > 2) {
            AF_ERROR("Sparse matmul can not be used in batch mode",
                     AF_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        af::dim4 ldims = lhsBase.dims();
        int lColDim    = (optLhs == AF_MAT_NONE) ? 1 : 0;
        int rRowDim    = (optRhs == AF_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, ldims[lColDim] == rhsInfo.dims()[rRowDim]);

        af_array output = 0;
        switch (lhs_type) {
            case f32:
                output = sparseMatmul<float>(lhs, rhs, optLhs, optRhs);
                break;
            case c32:
                output = sparseMatmul<cfloat>(lhs, rhs, optLhs, optRhs);
                break;
            case f64:
                output = sparseMatmul<double>(lhs, rhs, optLhs, optRhs);
                break;
            case c64:
                output = sparseMatmul<cdouble>(lhs, rhs, optLhs, optRhs);
                break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_matmul(af_array *out, const af_array lhs, const af_array rhs,
                 const af_mat_prop optLhs, const af_mat_prop optRhs) {
    using namespace detail;

    try {
        const ArrayInfo &lhsInfo = getInfo(lhs, false, true);
        const ArrayInfo &rhsInfo = getInfo(rhs, true, true);

        if (lhsInfo.isSparse())
            return af_sparse_matmul(out, lhs, rhs, optLhs, optRhs);

        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        if (!(optLhs == AF_MAT_NONE || optLhs == AF_MAT_TRANS ||
              optLhs == AF_MAT_CTRANS)) {
            AF_ERROR("Using this property is not yet supported in matmul",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (!(optRhs == AF_MAT_NONE || optRhs == AF_MAT_TRANS ||
              optRhs == AF_MAT_CTRANS)) {
            AF_ERROR("Using this property is not yet supported in matmul",
                     AF_ERR_NOT_SUPPORTED);
        }

        dim4 lDims = lhsInfo.dims();
        dim4 rDims = rhsInfo.dims();

        if (lDims.ndims() > 2 && rDims.ndims() > 2) {
            DIM_ASSERT(1, lDims.ndims() == rDims.ndims());
            if (lDims[2] != rDims[2] && lDims[2] != 1 && rDims[2] != 1) {
                AF_ERROR("Batch size mismatch along dimension 2", AF_ERR_BATCH);
            }
            if (lDims[3] != rDims[3] && lDims[3] != 1 && rDims[3] != 1) {
                AF_ERROR("Batch size mismatch along dimension 3", AF_ERR_BATCH);
            }
        }

        TYPE_ASSERT(lhs_type == rhs_type);
        af_array output = 0;

        int aColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
        int bRowDim = (optRhs == AF_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, lhsInfo.dims()[aColDim] == rhsInfo.dims()[bRowDim]);

        switch (lhs_type) {
            case f32: output = matmul<float>(lhs, rhs, optLhs, optRhs); break;
            case c32: output = matmul<cfloat>(lhs, rhs, optLhs, optRhs); break;
            case f64: output = matmul<double>(lhs, rhs, optLhs, optRhs); break;
            case c64: output = matmul<cdouble>(lhs, rhs, optLhs, optRhs); break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_dot(af_array *out, const af_array lhs, const af_array rhs,
              const af_mat_prop optLhs, const af_mat_prop optRhs) {
    using namespace detail;

    try {
        const ArrayInfo &lhsInfo = getInfo(lhs);
        const ArrayInfo &rhsInfo = getInfo(rhs);

        if (optLhs != AF_MAT_NONE && optLhs != AF_MAT_CONJ) {
            AF_ERROR("Using this property is not yet supported in dot",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (optRhs != AF_MAT_NONE && optRhs != AF_MAT_CONJ) {
            AF_ERROR("Using this property is not yet supported in dot",
                     AF_ERR_NOT_SUPPORTED);
        }

        DIM_ASSERT(1, lhsInfo.dims()[0] == rhsInfo.dims()[0]);
        af_dtype lhs_type = lhsInfo.getType();
        af_dtype rhs_type = rhsInfo.getType();

        if (lhsInfo.ndims() == 0) { return af_retain_array(out, lhs); }
        if (lhsInfo.ndims() > 1 || rhsInfo.ndims() > 1) {
            AF_ERROR("dot can not be used in batch mode", AF_ERR_BATCH);
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        af_array output = 0;

        switch (lhs_type) {
            case f32: output = dot<float>(lhs, rhs, optLhs, optRhs); break;
            case c32: output = dot<cfloat>(lhs, rhs, optLhs, optRhs); break;
            case f64: output = dot<double>(lhs, rhs, optLhs, optRhs); break;
            case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs); break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline T dotAll(af_array out) {
    T res;
    AF_CHECK(af_eval(out));
    AF_CHECK(af_get_data_ptr((void *)&res, out));
    return res;
}

af_err af_dot_all(double *rval, double *ival, const af_array lhs,
                  const af_array rhs, const af_mat_prop optLhs,
                  const af_mat_prop optRhs) {
    using namespace detail;

    try {
        *rval = 0;
        if (ival) *ival = 0;

        af_array out = 0;
        AF_CHECK(af_dot(&out, lhs, rhs, optLhs, optRhs));

        ArrayInfo lhsInfo = getInfo(lhs);
        af_dtype lhs_type = lhsInfo.getType();

        switch (lhs_type) {
            case f32: *rval = dotAll<float>(out); break;
            case f64: *rval = dotAll<double>(out); break;
            case c32: {
                cfloat temp = dotAll<cfloat>(out);
                *rval       = real(temp);
                if (ival) *ival = imag(temp);
            } break;
            case c64: {
                cdouble temp = dotAll<cdouble>(out);
                *rval        = real(temp);
                if (ival) *ival = imag(temp);
            } break;
            default: TYPE_ERROR(1, lhs_type);
        }

        if (out != 0) AF_CHECK(af_release_array(out));
    }
    CATCHALL
    return AF_SUCCESS;
}
