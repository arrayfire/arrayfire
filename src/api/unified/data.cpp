/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include "symbol_manager.hpp"

af_err af_constant(af_array *result, const double value, const unsigned ndims,
                   const dim_t *const dims, const af_dtype type) {
    CALL(af_constant, result, value, ndims, dims, type);
}

af_err af_constant_complex(af_array *arr, const double real, const double imag,
                           const unsigned ndims, const dim_t *const dims,
                           const af_dtype type) {
    CALL(af_constant_complex, arr, real, imag, ndims, dims, type);
}

af_err af_constant_long(af_array *arr, const long long val,
                        const unsigned ndims, const dim_t *const dims) {
    CALL(af_constant_long, arr, val, ndims, dims);
}

af_err af_constant_ulong(af_array *arr, const unsigned long long val,
                         const unsigned ndims, const dim_t *const dims) {
    CALL(af_constant_ulong, arr, val, ndims, dims);
}

af_err af_range(af_array *out, const unsigned ndims, const dim_t *const dims,
                const int seq_dim, const af_dtype type) {
    CALL(af_range, out, ndims, dims, seq_dim, type);
}

af_err af_iota(af_array *out, const unsigned ndims, const dim_t *const dims,
               const unsigned t_ndims, const dim_t *const tdims,
               const af_dtype type) {
    CALL(af_iota, out, ndims, dims, t_ndims, tdims, type);
}

af_err af_identity(af_array *out, const unsigned ndims, const dim_t *const dims,
                   const af_dtype type) {
    CALL(af_identity, out, ndims, dims, type);
}

af_err af_diag_create(af_array *out, const af_array in, const int num) {
    CHECK_ARRAYS(in);
    CALL(af_diag_create, out, in, num);
}

af_err af_diag_extract(af_array *out, const af_array in, const int num) {
    CHECK_ARRAYS(in);
    CALL(af_diag_extract, out, in, num);
}

af_err af_join(af_array *out, const int dim, const af_array first,
               const af_array second) {
    CHECK_ARRAYS(first, second);
    CALL(af_join, out, dim, first, second);
}

af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays,
                    const af_array *inputs) {
    for (unsigned i = 0; i < n_arrays; i++) { CHECK_ARRAYS(inputs[i]); }
    CALL(af_join_many, out, dim, n_arrays, inputs);
}

af_err af_tile(af_array *out, const af_array in, const unsigned x,
               const unsigned y, const unsigned z, const unsigned w) {
    CHECK_ARRAYS(in);
    CALL(af_tile, out, in, x, y, z, w);
}

af_err af_reorder(af_array *out, const af_array in, const unsigned x,
                  const unsigned y, const unsigned z, const unsigned w) {
    CHECK_ARRAYS(in);
    CALL(af_reorder, out, in, x, y, z, w);
}

af_err af_shift(af_array *out, const af_array in, const int x, const int y,
                const int z, const int w) {
    CHECK_ARRAYS(in);
    CALL(af_shift, out, in, x, y, z, w);
}

af_err af_moddims(af_array *out, const af_array in, const unsigned ndims,
                  const dim_t *const dims) {
    CHECK_ARRAYS(in);
    CALL(af_moddims, out, in, ndims, dims);
}

af_err af_flat(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_flat, out, in);
}

af_err af_flip(af_array *out, const af_array in, const unsigned dim) {
    CHECK_ARRAYS(in);
    CALL(af_flip, out, in, dim);
}

af_err af_lower(af_array *out, const af_array in, bool is_unit_diag) {
    CHECK_ARRAYS(in);
    CALL(af_lower, out, in, is_unit_diag);
}

af_err af_upper(af_array *out, const af_array in, bool is_unit_diag) {
    CHECK_ARRAYS(in);
    CALL(af_upper, out, in, is_unit_diag);
}

af_err af_select(af_array *out, const af_array cond, const af_array a,
                 const af_array b) {
    CHECK_ARRAYS(cond, a, b);
    CALL(af_select, out, cond, a, b);
}

af_err af_select_scalar_r(af_array *out, const af_array cond, const af_array a,
                          const double b) {
    CHECK_ARRAYS(cond, a);
    CALL(af_select_scalar_r, out, cond, a, b);
}

af_err af_select_scalar_l(af_array *out, const af_array cond, const double a,
                          const af_array b) {
    CHECK_ARRAYS(cond, b);
    CALL(af_select_scalar_l, out, cond, a, b);
}

af_err af_replace(af_array a, const af_array cond, const af_array b) {
    CHECK_ARRAYS(a, cond, b);
    CALL(af_replace, a, cond, b);
}

af_err af_replace_scalar(af_array a, const af_array cond, const double b) {
    CHECK_ARRAYS(a, cond);
    CALL(af_replace_scalar, a, cond, b);
}

af_err af_pad(af_array *out, const af_array in, const unsigned b_ndims,
              const dim_t *const b_dims, const unsigned e_ndims,
              const dim_t *const e_dims, const af_border_type ptype) {
    CHECK_ARRAYS(in);
    CALL(af_pad, out, in, b_ndims, b_dims, e_ndims, e_dims, ptype);
}
