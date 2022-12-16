/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/blas.h>

#include <Array.hpp>
#include <backend.hpp>
#include <blas.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <sparse_blas.hpp>
#include <sparse_handle.hpp>

#include <type_util.hpp>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>

using arrayfire::getSparseArray;
using arrayfire::getSparseArrayBase;
using arrayfire::common::half;
using arrayfire::common::SparseArrayBase;
using detail::cdouble;
using detail::cfloat;
using detail::gemm;
using detail::matmul;

namespace {
template<typename T>
static inline af_array sparseMatmul(const af_array lhs, const af_array rhs,
                                    af_mat_prop optLhs, af_mat_prop optRhs) {
    return getHandle(
        matmul<T>(getSparseArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline void gemm(af_array *out, af_mat_prop optLhs, af_mat_prop optRhs,
                        const T *alpha, const af_array lhs, const af_array rhs,
                        const T *betas) {
    gemm<T>(getArray<T>(*out), optLhs, optRhs, alpha, getArray<T>(lhs),
            getArray<T>(rhs), betas);
}

template<typename T>
static inline af_array dot(const af_array lhs, const af_array rhs,
                           af_mat_prop optLhs, af_mat_prop optRhs) {
    return getHandle(
        dot<T>(getArray<T>(lhs), getArray<T>(rhs), optLhs, optRhs));
}

template<typename T>
static inline T dotAll(af_array out) {
    T res{};
    AF_CHECK(af_eval(out));
    AF_CHECK(af_get_data_ptr((void *)&res, out));
    return res;
}

}  // namespace

af_err af_sparse_matmul(af_array *out, const af_array lhs, const af_array rhs,
                        const af_mat_prop optLhs, const af_mat_prop optRhs) {
    try {
        const SparseArrayBase lhsBase = getSparseArrayBase(lhs);
        const ArrayInfo &rhsInfo      = getInfo(rhs);

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

af_err af_gemm(af_array *out, const af_mat_prop optLhs,
               const af_mat_prop optRhs, const void *alpha, const af_array lhs,
               const af_array rhs, const void *beta) {
    try {
        const ArrayInfo &lhsInfo = getInfo(lhs, false, true);
        const ArrayInfo &rhsInfo = getInfo(rhs, true, true);

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

        af::dim4 lDims = lhsInfo.dims();
        af::dim4 rDims = rhsInfo.dims();

        if (lDims.ndims() > 2 && rDims.ndims() > 2) {
            DIM_ASSERT(3, lDims.ndims() == rDims.ndims());
            if (lDims[2] != rDims[2] && lDims[2] != 1 && rDims[2] != 1) {
                AF_ERROR("Batch size mismatch along dimension 2", AF_ERR_BATCH);
            }
            if (lDims[3] != rDims[3] && lDims[3] != 1 && rDims[3] != 1) {
                AF_ERROR("Batch size mismatch along dimension 3", AF_ERR_BATCH);
            }
        }

        TYPE_ASSERT(lhs_type == rhs_type);

        int aColDim = (optLhs == AF_MAT_NONE) ? 1 : 0;
        int bRowDim = (optRhs == AF_MAT_NONE) ? 0 : 1;

        DIM_ASSERT(1, lhsInfo.dims()[aColDim] == rhsInfo.dims()[bRowDim]);

        // Assume that *out is either initialized to null or an actual af_array
        // Otherwise, this function has undefined behavior
        af_array output = 0;
        if (*out) {
            output = *out;
        } else {
            const int aRowDim    = (optLhs == AF_MAT_NONE) ? 0 : 1;
            const int bColDim    = (optRhs == AF_MAT_NONE) ? 1 : 0;
            const int M          = lDims[aRowDim];
            const int N          = rDims[bColDim];
            const dim_t d2       = std::max(lDims[2], rDims[2]);
            const dim_t d3       = std::max(lDims[3], rDims[3]);
            const af::dim4 oDims = af::dim4(M, N, d2, d3);
            AF_CHECK(af_create_handle(&output, lhsInfo.ndims(), oDims.get(),
                                      lhs_type));
        }

        switch (lhs_type) {
            case f32:
                gemm<float>(&output, optLhs, optRhs,
                            static_cast<const float *>(alpha), lhs, rhs,
                            static_cast<const float *>(beta));
                break;
            case c32:
                gemm<cfloat>(&output, optLhs, optRhs,
                             static_cast<const cfloat *>(alpha), lhs, rhs,
                             static_cast<const cfloat *>(beta));
                break;
            case f64:
                gemm<double>(&output, optLhs, optRhs,
                             static_cast<const double *>(alpha), lhs, rhs,
                             static_cast<const double *>(beta));
                break;
            case c64:
                gemm<cdouble>(&output, optLhs, optRhs,
                              static_cast<const cdouble *>(alpha), lhs, rhs,
                              static_cast<const cdouble *>(beta));
                break;
            case f16:
                gemm<half>(&output, optLhs, optRhs,
                           static_cast<const half *>(alpha), lhs, rhs,
                           static_cast<const half *>(beta));
                break;
            default: TYPE_ERROR(3, lhs_type);
        }

        std::swap(*out, output);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_matmul(af_array *out, const af_array lhs, const af_array rhs,
                 const af_mat_prop optLhs, const af_mat_prop optRhs) {
    try {
        const ArrayInfo &lhsInfo = getInfo(lhs, false, true);
        const ArrayInfo &rhsInfo = getInfo(rhs, true, true);

        if (lhsInfo.isSparse()) {
            return af_sparse_matmul(out, lhs, rhs, optLhs, optRhs);
        }

        const int aRowDim = (optLhs == AF_MAT_NONE) ? 0 : 1;
        const int bColDim = (optRhs == AF_MAT_NONE) ? 1 : 0;

        const af::dim4 &lDims = lhsInfo.dims();
        const af::dim4 &rDims = rhsInfo.dims();
        const int M           = lDims[aRowDim];
        const int N           = rDims[bColDim];

        const dim_t d2       = std::max(lDims[2], rDims[2]);
        const dim_t d3       = std::max(lDims[3], rDims[3]);
        const af::dim4 oDims = af::dim4(M, N, d2, d3);

        af_array gemm_out = 0;
        AF_CHECK(af_create_handle(&gemm_out, oDims.ndims(), oDims.get(),
                                  lhsInfo.getType()));

        af_dtype lhs_type = lhsInfo.getType();
        switch (lhs_type) {
            case f16: {
                static const half alpha(1.0f);
                static const half beta(0.0f);
                AF_CHECK(af_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case f32: {
                float alpha = 1.f;
                float beta  = 0.f;
                AF_CHECK(af_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case c32: {
                cfloat alpha{1.f, 0.f};
                cfloat beta{0.f, 0.f};

                AF_CHECK(af_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case f64: {
                double alpha = 1.0;
                double beta  = 0.0;
                AF_CHECK(af_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            case c64: {
                cdouble alpha{1.0, 0.0};
                cdouble beta{0.0, 0.0};
                AF_CHECK(af_gemm(&gemm_out, optLhs, optRhs, &alpha, lhs, rhs,
                                 &beta));
                break;
            }
            default: TYPE_ERROR(1, lhs_type);
        }

        std::swap(*out, gemm_out);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_dot(af_array *out, const af_array lhs, const af_array rhs,
              const af_mat_prop optLhs, const af_mat_prop optRhs) {
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
            case f16: output = dot<half>(lhs, rhs, optLhs, optRhs); break;
            case f32: output = dot<float>(lhs, rhs, optLhs, optRhs); break;
            case c32: output = dot<cfloat>(lhs, rhs, optLhs, optRhs); break;
            case f64: output = dot<double>(lhs, rhs, optLhs, optRhs); break;
            case c64: output = dot<cdouble>(lhs, rhs, optLhs, optRhs); break;
            default: TYPE_ERROR(1, lhs_type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_dot_all(double *rval, double *ival, const af_array lhs,
                  const af_array rhs, const af_mat_prop optLhs,
                  const af_mat_prop optRhs) {
    using namespace detail;  // NOLINT needed for imag and real functions
                             // name resolution

    try {
        *rval = 0;
        if (ival) { *ival = 0; }

        af_array out = 0;
        AF_CHECK(af_dot(&out, lhs, rhs, optLhs, optRhs));

        const ArrayInfo &lhsInfo = getInfo(lhs);
        af_dtype lhs_type        = lhsInfo.getType();

        switch (lhs_type) {
            case f16: *rval = static_cast<double>(dotAll<half>(out)); break;
            case f32: *rval = dotAll<float>(out); break;
            case f64: *rval = dotAll<double>(out); break;
            case c32: {
                cfloat temp = dotAll<cfloat>(out);
                *rval       = real(temp);
                if (ival) { *ival = imag(temp); }
            } break;
            case c64: {
                cdouble temp = dotAll<cdouble>(out);
                *rval        = real(temp);
                if (ival) { *ival = imag(temp); }
            } break;
            default: TYPE_ERROR(1, lhs_type);
        }

        if (out != 0) { AF_CHECK(af_release_array(out)); }
    }
    CATCHALL
    return AF_SUCCESS;
}
