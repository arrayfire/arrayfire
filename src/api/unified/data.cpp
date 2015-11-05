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

af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_t * const dims,
                   const af_dtype type)
{
    return CALL(result, value, ndims, dims, type);
}


af_err af_constant_complex(af_array *arr, const double real, const double imag,
        const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(arr, real, imag, ndims, dims, type);
}


af_err af_constant_long (af_array *arr, const  intl val, const unsigned ndims, const dim_t * const dims)
{
    return CALL(arr, val, ndims, dims);
}


af_err af_constant_ulong(af_array *arr, const uintl val, const unsigned ndims, const dim_t * const dims)
{
    return CALL(arr, val, ndims, dims);
}

af_err af_range(af_array *out, const unsigned ndims, const dim_t * const dims,
        const int seq_dim, const af_dtype type)
{
    return CALL(out, ndims, dims, seq_dim, type);
}

af_err af_iota(af_array *out, const unsigned ndims, const dim_t * const dims,
        const unsigned t_ndims, const dim_t * const tdims, const af_dtype type)
{
    return CALL(out, ndims, dims, t_ndims, tdims, type);
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(out, ndims, dims, type);
}

af_err af_randu_gen(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type,
        const af_random_type rtype)
{
    return CALL(out, ndims, dims, type, rtype);
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(out, ndims, dims, type);
}

af_err af_set_seed(const uintl seed)
{
    return CALL(seed);
}

af_err af_get_seed(uintl *seed)
{
    return CALL(seed);
}

af_err af_identity(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    return CALL(out, ndims, dims, type);
}

af_err af_diag_create(af_array *out, const af_array in, const int num)
{
    return CALL(out, in, num);
}

af_err af_diag_extract(af_array *out, const af_array in, const int num)
{
    return CALL(out, in, num);
}

af_err af_join(af_array *out, const int dim, const af_array first, const af_array second)
{
    return CALL(out, dim, first, second);
}

af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs)
{
    return CALL(out, dim, n_arrays, inputs);
}

af_err af_tile(af_array *out, const af_array in,
        const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    return CALL(out, in, x, y, z, w);
}

af_err af_reorder(af_array *out, const af_array in,
        const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    return CALL(out, in, x, y, z, w);
}

af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w)
{
    return CALL(out, in, x, y, z, w);
}

af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_t * const dims)
{
    return CALL(out, in, ndims, dims);
}

af_err af_flat(af_array *out, const af_array in)
{
    return CALL(out, in);
}

af_err af_flip(af_array *out, const af_array in, const unsigned dim)
{
    return CALL(out, in, dim);
}

af_err af_lower(af_array *out, const af_array in, bool is_unit_diag)
{
    return CALL(out, in, is_unit_diag);
}

af_err af_upper(af_array *out, const af_array in, bool is_unit_diag)
{
    return CALL(out, in, is_unit_diag);
}

af_err af_select(af_array *out, const af_array cond, const af_array a, const af_array b)
{
    return CALL(out, cond, a, b);
}

af_err af_select_scalar_r(af_array *out, const af_array cond, const af_array a, const double b)
{
    return CALL(out, cond, a, b);
}

af_err af_select_scalar_l(af_array *out, const af_array cond, const double a, const af_array b)
{
    return CALL(out, cond, a, b);
}

af_err af_replace(af_array a, const af_array cond, const af_array b)
{
    return CALL(a, cond, b);
}

af_err af_replace_scalar(af_array a, const af_array cond, const double b)
{
    return CALL(a, cond, b);
}
