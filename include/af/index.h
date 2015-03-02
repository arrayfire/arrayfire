/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <af/defines.h>

#if __cplusplus
namespace af
{

AFAPI array join(const int dim, const array &first, const array &second);

AFAPI array tile(const array& in, const unsigned x, const unsigned y=1, const unsigned z=1, const unsigned w=1);

AFAPI array reorder(const array& in, const unsigned x, const unsigned y=1, const unsigned z=2, const unsigned w=3);

AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);

AFAPI array moddims(const array& in, const unsigned ndims, const dim_type * const dims);

AFAPI array moddims(const array& in, const dim4& dims);

AFAPI array moddims(const array& in, dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1);

AFAPI array flat(const array &in);

AFAPI array flip(const array &in, const unsigned dim);

AFAPI array lookup(const array &in, const array &idx, const int dim = -1);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Create a new af_array by indexing from existing af_array.
    // This takes the form `out = in(seq_a, seq_b)`
    AFAPI af_err af_index(af_array *out, const af_array in, unsigned ndims, const af_seq* const index);

    // create a new af_array by indexing existing af_array using another af_array
    AFAPI af_err af_lookup(af_array *out, const af_array in, const af_array indices, const unsigned dim);

    // copy an array into exiting array of larger dimensions
    // error out in case of insufficient dimension lengths
    AFAPI af_err af_assign(af_array *out, const af_array lhs, unsigned ndims, const af_seq* const index, const af_array rhs);

    // Join 2 Arrays
    AFAPI af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);

    // Tile an Array
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    // re-shape the the dimensions of the input array
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);

    AFAPI af_err af_flat(af_array *out, const af_array in);

    AFAPI af_err af_flip(af_array *out, const af_array in, const unsigned dim);

#ifdef __cplusplus
}
#endif
