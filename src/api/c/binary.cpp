/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/moddims.hpp>
#include <common/tile.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <optypes.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>

#include <arith.hpp>
#include <logic.hpp>
#include <sparse_arith.hpp>

#include <common/half.hpp>

using af::dim4;
using af::dtype;
using arrayfire::castSparse;
using arrayfire::getSparseArray;
using arrayfire::getSparseArrayBase;
using arrayfire::common::half;
using arrayfire::common::modDims;
using arrayfire::common::SparseArrayBase;
using arrayfire::common::tile;
using detail::arithOp;
using detail::arithOpD;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, af_op_t op>
static inline af_array arithOp(const af_array lhs, const af_array rhs,
                               const dim4 &odims) {
    const ArrayInfo &linfo = getInfo(lhs);
    const ArrayInfo &rinfo = getInfo(rhs);

    dtype type = static_cast<af::dtype>(af::dtype_traits<T>::af_type);

    const detail::Array<T> &l =
        linfo.getType() == type ? getArray<T>(lhs) : castArray<T>(lhs);
    const detail::Array<T> &r =
        rinfo.getType() == type ? getArray<T>(rhs) : castArray<T>(rhs);

    return getHandle(arithOp<T, op>(l, r, odims));
}

template<typename T, af_op_t op>
static inline af_array arithOpBroadcast(const af_array lhs,
                                        const af_array rhs) {
    const ArrayInfo &linfo = getInfo(lhs);
    const ArrayInfo &rinfo = getInfo(rhs);

    dim4 odims(1), ltile(1), rtile(1);
    dim4 lshape = linfo.dims();
    dim4 rshape = rinfo.dims();

    for (int d = 0; d < AF_MAX_DIMS; ++d) {
        DIM_ASSERT(
            1, ((lshape[d] == rshape[d]) || (lshape[d] == 1 && rshape[d] > 1) ||
                (lshape[d] > 1 && rshape[d] == 1)));
        odims[d] = std::max(lshape[d], rshape[d]);
        if (lshape[d] == rshape[d]) {
            ltile[d] = rtile[d] = 1;
        } else if (lshape[d] == 1 && rshape[d] > 1) {
            ltile[d] = odims[d];
        } else if (lshape[d] > 1 && rshape[d] == 1) {
            rtile[d] = odims[d];
        }
    }

    Array<T> lhst =
        arrayfire::common::tile<T>(modDims(getArray<T>(lhs), lshape), ltile);
    Array<T> rhst =
        arrayfire::common::tile<T>(modDims(getArray<T>(rhs), rshape), rtile);

    return getHandle(arithOp<T, op>(lhst, rhst, odims));
}

template<typename T, af_op_t op>
static inline af_array sparseArithOp(const af_array lhs, const af_array rhs) {
    auto res = arithOp<T, op>(getSparseArray<T>(lhs), getSparseArray<T>(rhs));
    return getHandle(res);
}

template<typename T, af_op_t op>
static inline af_array arithSparseDenseOp(const af_array lhs,
                                          const af_array rhs,
                                          const bool reverse) {
    if (op == af_add_t || op == af_sub_t) {
        return getHandle(
            arithOpD<T, op>(castSparse<T>(lhs), castArray<T>(rhs), reverse));
    }
    if (op == af_mul_t || op == af_div_t) {
        return getHandle(
            arithOp<T, op>(castSparse<T>(lhs), castArray<T>(rhs), reverse));
    }
}

template<af_op_t op>
static af_err af_arith(af_array *out, const af_array lhs, const af_array rhs,
                       const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        const af_dtype otype = implicit(linfo.getType(), rinfo.getType());
        af_array res;

        if (batchMode || linfo.dims() == rinfo.dims()) {
            dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
            if (odims.ndims() == 0) {
                return af_create_handle(out, 0, nullptr, otype);
            }

            switch (otype) {
                case f32: res = arithOp<float, op>(lhs, rhs, odims); break;
                case f64: res = arithOp<double, op>(lhs, rhs, odims); break;
                case c32: res = arithOp<cfloat, op>(lhs, rhs, odims); break;
                case c64: res = arithOp<cdouble, op>(lhs, rhs, odims); break;
                case s32: res = arithOp<int, op>(lhs, rhs, odims); break;
                case u32: res = arithOp<uint, op>(lhs, rhs, odims); break;
                case u8: res = arithOp<uchar, op>(lhs, rhs, odims); break;
                case b8: res = arithOp<char, op>(lhs, rhs, odims); break;
                case s64: res = arithOp<intl, op>(lhs, rhs, odims); break;
                case u64: res = arithOp<uintl, op>(lhs, rhs, odims); break;
                case s16: res = arithOp<short, op>(lhs, rhs, odims); break;
                case u16: res = arithOp<ushort, op>(lhs, rhs, odims); break;
                case f16: res = arithOp<half, op>(lhs, rhs, odims); break;
                default: TYPE_ERROR(0, otype);
            }
        } else {
            if (linfo.ndims() == 0 && rinfo.ndims() == 0) {
                return af_create_handle(out, 0, nullptr, otype);
            }
            switch (otype) {
                case f32: res = arithOpBroadcast<float, op>(lhs, rhs); break;
                case f64: res = arithOpBroadcast<double, op>(lhs, rhs); break;
                case c32: res = arithOpBroadcast<cfloat, op>(lhs, rhs); break;
                case c64: res = arithOpBroadcast<cdouble, op>(lhs, rhs); break;
                case s32: res = arithOpBroadcast<int, op>(lhs, rhs); break;
                case u32: res = arithOpBroadcast<uint, op>(lhs, rhs); break;
                case u8: res = arithOpBroadcast<uchar, op>(lhs, rhs); break;
                case b8: res = arithOpBroadcast<char, op>(lhs, rhs); break;
                case s64: res = arithOpBroadcast<intl, op>(lhs, rhs); break;
                case u64: res = arithOpBroadcast<uintl, op>(lhs, rhs); break;
                case s16: res = arithOpBroadcast<short, op>(lhs, rhs); break;
                case u16: res = arithOpBroadcast<ushort, op>(lhs, rhs); break;
                case f16: res = arithOpBroadcast<half, op>(lhs, rhs); break;
                default: TYPE_ERROR(0, otype);
            }
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_arith_real(af_array *out, const af_array lhs,
                            const af_array rhs, const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
        const af_dtype otype = implicit(linfo.getType(), rinfo.getType());
        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, otype);
        }

        af_array res;
        switch (otype) {
            case f32: res = arithOp<float, op>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, op>(lhs, rhs, odims); break;
            case s32: res = arithOp<int, op>(lhs, rhs, odims); break;
            case u32: res = arithOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = arithOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = arithOp<char, op>(lhs, rhs, odims); break;
            case s64: res = arithOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = arithOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = arithOp<short, op>(lhs, rhs, odims); break;
            case u16: res = arithOp<ushort, op>(lhs, rhs, odims); break;
            case f16: res = arithOp<half, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, otype);
        }
        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_arith_sparse(af_array *out, const af_array lhs,
                              const af_array rhs) {
    try {
        const SparseArrayBase linfo = getSparseArrayBase(lhs);
        const SparseArrayBase rinfo = getSparseArrayBase(rhs);

        ARG_ASSERT(1, (linfo.getStorage() == rinfo.getStorage()));
        ARG_ASSERT(1, (linfo.dims() == rinfo.dims()));
        ARG_ASSERT(1, (linfo.getStorage() == AF_STORAGE_CSR));

        const af_dtype otype = implicit(linfo.getType(), rinfo.getType());
        af_array res;
        switch (otype) {
            case f32: res = sparseArithOp<float, op>(lhs, rhs); break;
            case f64: res = sparseArithOp<double, op>(lhs, rhs); break;
            case c32: res = sparseArithOp<cfloat, op>(lhs, rhs); break;
            case c64: res = sparseArithOp<cdouble, op>(lhs, rhs); break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<af_op_t op>
static af_err af_arith_sparse_dense(af_array *out, const af_array lhs,
                                    const af_array rhs,
                                    const bool reverse = false) {
    try {
        const SparseArrayBase linfo = getSparseArrayBase(lhs);
        if (linfo.ndims() > 2) {
            AF_ERROR(
                "Sparse-Dense arithmetic operations cannot be used in batch "
                "mode",
                AF_ERR_BATCH);
        }
        const ArrayInfo &rinfo = getInfo(rhs);

        const af_dtype otype = implicit(linfo.getType(), rinfo.getType());
        af_array res;
        switch (otype) {
            case f32:
                res = arithSparseDenseOp<float, op>(lhs, rhs, reverse);
                break;
            case f64:
                res = arithSparseDenseOp<double, op>(lhs, rhs, reverse);
                break;
            case c32:
                res = arithSparseDenseOp<cfloat, op>(lhs, rhs, reverse);
                break;
            case c64:
                res = arithSparseDenseOp<cdouble, op>(lhs, rhs, reverse);
                break;
            default: TYPE_ERROR(0, otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_add(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            return af_arith_sparse<af_add_t>(out, lhs, rhs);
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return af_arith_sparse_dense<af_add_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            // second operand(Array) of af_arith call should be dense
            return af_arith_sparse_dense<af_add_t>(out, rhs, lhs, true);
        }
        return af_arith<af_add_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

af_err af_mul(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            // return af_arith_sparse<af_mul_t>(out, lhs, rhs);
            // MKL doesn't have mul or div support yet, hence
            // this is commented out although alternative cpu code exists
            return AF_ERR_NOT_SUPPORTED;
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return af_arith_sparse_dense<af_mul_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            return af_arith_sparse_dense<af_mul_t>(
                out, rhs, lhs,
                true);  // dense should be rhs
        }
        return af_arith<af_mul_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

af_err af_sub(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            return af_arith_sparse<af_sub_t>(out, lhs, rhs);
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return af_arith_sparse_dense<af_sub_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            return af_arith_sparse_dense<af_sub_t>(
                out, rhs, lhs,
                true);  // dense should be rhs
        }
        return af_arith<af_sub_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

af_err af_div(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    try {
        // Check if inputs are sparse
        const ArrayInfo &linfo = getInfo(lhs, false);
        const ArrayInfo &rinfo = getInfo(rhs, false);

        if (linfo.isSparse() && rinfo.isSparse()) {
            // return af_arith_sparse<af_div_t>(out, lhs, rhs);
            // MKL doesn't have mul or div support yet, hence
            // this is commented out although alternative cpu code exists
            return AF_ERR_NOT_SUPPORTED;
        }
        if (linfo.isSparse() && !rinfo.isSparse()) {
            return af_arith_sparse_dense<af_div_t>(out, lhs, rhs);
        }
        if (!linfo.isSparse() && rinfo.isSparse()) {
            // Division by sparse is currently not allowed - for convinence of
            // dealing with division by 0
            // return af_arith_sparse_dense<af_div_t>(out, rhs, lhs, true); //
            // dense should be rhs
            return AF_ERR_NOT_SUPPORTED;
        }
        return af_arith<af_div_t>(out, lhs, rhs, batchMode);
    }
    CATCHALL;
}

af_err af_maxof(af_array *out, const af_array lhs, const af_array rhs,
                const bool batchMode) {
    return af_arith<af_max_t>(out, lhs, rhs, batchMode);
}

af_err af_minof(af_array *out, const af_array lhs, const af_array rhs,
                const bool batchMode) {
    return af_arith<af_min_t>(out, lhs, rhs, batchMode);
}

af_err af_rem(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    return af_arith_real<af_rem_t>(out, lhs, rhs, batchMode);
}

af_err af_mod(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    return af_arith_real<af_mod_t>(out, lhs, rhs, batchMode);
}

af_err af_pow(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);
        if (rinfo.isComplex()) {
            af_array log_lhs, log_res;
            af_array res;
            AF_CHECK(af_log(&log_lhs, lhs));
            AF_CHECK(af_mul(&log_res, log_lhs, rhs, batchMode));
            AF_CHECK(af_exp(&res, log_res));
            AF_CHECK(af_release_array(log_lhs));
            AF_CHECK(af_release_array(log_res));
            std::swap(*out, res);
            return AF_SUCCESS;
        }
        if (linfo.isComplex()) {
            af_array mag, angle;
            af_array mag_res, angle_res;
            af_array real_res, imag_res, cplx_res;
            af_array res;
            AF_CHECK(af_abs(&mag, lhs));
            AF_CHECK(af_arg(&angle, lhs));
            AF_CHECK(af_pow(&mag_res, mag, rhs, batchMode));
            AF_CHECK(af_mul(&angle_res, angle, rhs, batchMode));
            AF_CHECK(af_cos(&real_res, angle_res));
            AF_CHECK(af_sin(&imag_res, angle_res));
            AF_CHECK(af_cplx2(&cplx_res, real_res, imag_res, batchMode));
            AF_CHECK(af_mul(&res, mag_res, cplx_res, batchMode));
            AF_CHECK(af_release_array(mag));
            AF_CHECK(af_release_array(angle));
            AF_CHECK(af_release_array(mag_res));
            AF_CHECK(af_release_array(angle_res));
            AF_CHECK(af_release_array(real_res));
            AF_CHECK(af_release_array(imag_res));
            AF_CHECK(af_release_array(cplx_res));
            std::swap(*out, res);
            return AF_SUCCESS;
        }
    }
    CATCHALL;

    return af_arith_real<af_pow_t>(out, lhs, rhs, batchMode);
}

af_err af_root(af_array *out, const af_array lhs, const af_array rhs,
               const bool batchMode) {
    try {
        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);
        if (linfo.isComplex() || rinfo.isComplex()) {
            af_array log_lhs, log_res;
            af_array res;
            AF_CHECK(af_log(&log_lhs, lhs));
            AF_CHECK(af_div(&log_res, log_lhs, rhs, batchMode));
            AF_CHECK(af_exp(&res, log_res));
            std::swap(*out, res);
            return AF_SUCCESS;
        }

        af_array one;
        AF_CHECK(af_constant(&one, 1, linfo.ndims(), linfo.dims().get(),
                             linfo.getType()));

        af_array inv_lhs;
        AF_CHECK(af_div(&inv_lhs, one, lhs, batchMode));

        AF_CHECK(af_arith_real<af_pow_t>(out, rhs, inv_lhs, batchMode));

        AF_CHECK(af_release_array(one));
        AF_CHECK(af_release_array(inv_lhs));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_atan2(af_array *out, const af_array lhs, const af_array rhs,
                const bool batchMode) {
    try {
        const af_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for atan2 ",
                     AF_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);
        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res;
        switch (type) {
            case f32: res = arithOp<float, af_atan2_t>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, af_atan2_t>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_hypot(af_array *out, const af_array lhs, const af_array rhs,
                const bool batchMode) {
    try {
        const af_dtype type = implicit(lhs, rhs);

        if (type != f32 && type != f64) {
            AF_ERROR("Only floating point arrays are supported for hypot ",
                     AF_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res;
        switch (type) {
            case f32: res = arithOp<float, af_hypot_t>(lhs, rhs, odims); break;
            case f64: res = arithOp<double, af_hypot_t>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T, af_op_t op>
static inline af_array logicOp(const af_array lhs, const af_array rhs,
                               const dim4 &odims) {
    af_array res =
        getHandle(logicOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<af_op_t op>
static af_err af_logic(af_array *out, const af_array lhs, const af_array rhs,
                       const bool batchMode) {
    try {
        const af_dtype type = implicit(lhs, rhs);

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res;
        switch (type) {
            case f32: res = logicOp<float, op>(lhs, rhs, odims); break;
            case f64: res = logicOp<double, op>(lhs, rhs, odims); break;
            case c32: res = logicOp<cfloat, op>(lhs, rhs, odims); break;
            case c64: res = logicOp<cdouble, op>(lhs, rhs, odims); break;
            case s32: res = logicOp<int, op>(lhs, rhs, odims); break;
            case u32: res = logicOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = logicOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = logicOp<char, op>(lhs, rhs, odims); break;
            case s64: res = logicOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = logicOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = logicOp<short, op>(lhs, rhs, odims); break;
            case u16: res = logicOp<ushort, op>(lhs, rhs, odims); break;
            case f16: res = logicOp<half, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_eq(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_eq_t>(out, lhs, rhs, batchMode);
}

af_err af_neq(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    return af_logic<af_neq_t>(out, lhs, rhs, batchMode);
}

af_err af_gt(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_gt_t>(out, lhs, rhs, batchMode);
}

af_err af_ge(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_ge_t>(out, lhs, rhs, batchMode);
}

af_err af_lt(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_lt_t>(out, lhs, rhs, batchMode);
}

af_err af_le(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_le_t>(out, lhs, rhs, batchMode);
}

af_err af_and(af_array *out, const af_array lhs, const af_array rhs,
              const bool batchMode) {
    return af_logic<af_and_t>(out, lhs, rhs, batchMode);
}

af_err af_or(af_array *out, const af_array lhs, const af_array rhs,
             const bool batchMode) {
    return af_logic<af_or_t>(out, lhs, rhs, batchMode);
}

template<typename T, af_op_t op>
static inline af_array bitOp(const af_array lhs, const af_array rhs,
                             const dim4 &odims) {
    af_array res =
        getHandle(bitOp<T, op>(castArray<T>(lhs), castArray<T>(rhs), odims));
    return res;
}

template<af_op_t op>
static af_err af_bitwise(af_array *out, const af_array lhs, const af_array rhs,
                         const bool batchMode) {
    try {
        const af_dtype type = implicit(lhs, rhs);

        const ArrayInfo &linfo = getInfo(lhs);
        const ArrayInfo &rinfo = getInfo(rhs);

        dim4 odims = getOutDims(linfo.dims(), rinfo.dims(), batchMode);

        if (odims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, type);
        }

        af_array res;
        switch (type) {
            case s32: res = bitOp<int, op>(lhs, rhs, odims); break;
            case u32: res = bitOp<uint, op>(lhs, rhs, odims); break;
            case u8: res = bitOp<uchar, op>(lhs, rhs, odims); break;
            case b8: res = bitOp<char, op>(lhs, rhs, odims); break;
            case s64: res = bitOp<intl, op>(lhs, rhs, odims); break;
            case u64: res = bitOp<uintl, op>(lhs, rhs, odims); break;
            case s16: res = bitOp<short, op>(lhs, rhs, odims); break;
            case u16: res = bitOp<ushort, op>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_bitand(af_array *out, const af_array lhs, const af_array rhs,
                 const bool batchMode) {
    return af_bitwise<af_bitand_t>(out, lhs, rhs, batchMode);
}

af_err af_bitor(af_array *out, const af_array lhs, const af_array rhs,
                const bool batchMode) {
    return af_bitwise<af_bitor_t>(out, lhs, rhs, batchMode);
}

af_err af_bitxor(af_array *out, const af_array lhs, const af_array rhs,
                 const bool batchMode) {
    return af_bitwise<af_bitxor_t>(out, lhs, rhs, batchMode);
}

af_err af_bitshiftl(af_array *out, const af_array lhs, const af_array rhs,
                    const bool batchMode) {
    return af_bitwise<af_bitshiftl_t>(out, lhs, rhs, batchMode);
}

af_err af_bitshiftr(af_array *out, const af_array lhs, const af_array rhs,
                    const bool batchMode) {
    return af_bitwise<af_bitshiftr_t>(out, lhs, rhs, batchMode);
}
