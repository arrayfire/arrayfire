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

/**
   \defgroup manip_func_join join
   @{
   Join two arrays along specified dimension

   \param[in] dim is the dimension along which join occurs
   \param[in] first is the first input array
   \param[in] second is the first second array
   \return the array that joins \p first and \p second along \p dim

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array join(const int dim, const array &first, const array &second);
/**
   @}
*/

/**
   \defgroup manip_func_tile tile
   @{
   Tile the input array along specified dimensions

   \param[in] in is the input matrix
   \param[in] x is the number of times \p in is tiled along first dimension
   \param[in] y is the number of times \p in is tiled along second dimension
   \param[in] z is the number of times \p in is tiled along third dimension
   \param[in] w is the number of times \p in is tiled along fourth dimension
   \return the tiled output

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array tile(const array &in, const unsigned x, const unsigned y=1, const unsigned z=1, const unsigned w=1);
/**
   @}
*/

/**
   \defgroup manip_func_tile tile
   @{
   Tile the input array along specified dimensions

   \param[in] in is the input matrix
   \param[in] dims dim4 of tile dimensions
   \return the tiled output

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array tile(const array &in, const af::dim4 &dims);
/**
   @}
*/

/**
   \defgroup manip_func_reorder reorder
   @{
   Reorder the input by in the specified order

   \param[in] in is the input matrix
   \param[in] x specifies which dimension should be first
   \param[in] y specifies which dimension should be second
   \param[in] z specifies which dimension should be third
   \param[in] w specifies which dimension should be fourth
   \return the reordered output

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array reorder(const array& in, const unsigned x, const unsigned y=1, const unsigned z=2, const unsigned w=3);
/**
   @}
*/

/**
   \defgroup manip_func_shift shift
   @{
   Circular shift slong specified dimensions

   \param[in] in is the input matrix
   \param[in] x specifies the shift along first dimension
   \param[in] y specifies the shift along second dimension
   \param[in] z specifies the shift along third dimension
   \param[in] w specifies the shift along fourth dimension
   \return the shifted output

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);
/**
   @}
*/

/**
   \defgroup manip_func_moddims moddims
   @{

   Modify the input dimensions without changing the data order

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array moddims(const array& in, const unsigned ndims, const dim_type * const dims);

AFAPI array moddims(const array& in, const dim4& dims);

AFAPI array moddims(const array& in, dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1);
/**
   @}
*/

/**
   \defgroup manip_func_flat flat
   @{

   Flatten the input to a single dimension

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array flat(const array &in);
/**
   @}
*/

/**
   \defgroup manip_func_flip flip
   @{

   Flip the input along sepcified dimension

   \ingroup manip_mat
   \ingroup arrayfire_func
*/
AFAPI array flip(const array &in, const unsigned dim);
/**
   @}
*/


/**
   Lookup the values of input array based on index

   \param[in] in is input lookup array
   \param[in] idx is lookup indices
   \param[in] dim specifies the dimension for indexing
   \returns an array containing values at locations specified by \p index

   \ingroup index_func_index
*/

AFAPI array lookup(const array &in, const array &idx, const int dim = -1);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   Lookup the values of input array based on sequences

   \param[out] out will contain an array containing values at indexed by the sequences
   \param[in] in is the input array
   \param[in] ndims is the number of sequences provided
   \param[in] index is an array of sequences

   \ingroup index_func_index
*/
    AFAPI af_err af_index(af_array *out, const af_array in, unsigned ndims, const af_seq* const index);


/**
   Lookup the values of input array based on index

   \param[out] out will contain an array containing values at locations specified by \p index
   \param[in] in is input lookup array
   \param[in] indices is lookup indices
   \param[in] dim specifies the dimension for indexing

   \ingroup index_func_index
*/
    AFAPI af_err af_lookup(af_array *out, const af_array in, const af_array indices, const unsigned dim);

/**
   Copy and write values in the locations specified by the sequences

   \param[out] out will contain an array with values of \p rhs copied to locations specified by \p index and values from \p lhs in all other locations.

   \param[in] lhs is array whose values are used for indices NOT specified by \p index
   \param[in] ndims is the number of sequences provided
   \param[in] index is an array of sequences
   \param[in] rhs is the array whose values are used for indices specified by \p index

   \ingroup index_func_assign
*/
    AFAPI af_err af_assign_seq(af_array *out, const af_array lhs, unsigned ndims, const af_seq* const index, const af_array rhs);

    /**
       \ingroup manip_func_join
    */
    AFAPI af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);

    /**
       \ingroup manip_func_tile
    */
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       \ingroup manip_func_reorder
    */
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       \ingroup manip_func_shift
    */
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    /**
       \ingroup manip_func_moddims
    */
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);

    /**
       \ingroup manip_func_flat
    */
    AFAPI af_err af_flat(af_array *out, const af_array in);

    /**
       \ingroup manip_func_flip
    */
    AFAPI af_err af_flip(af_array *out, const af_array in, const unsigned dim);

    // generalized indexing function that accepts either af_array or af_seq
    // along a dimension to index the input array and create the corresponding
    // output array
    AFAPI af_err af_index_gen(af_array *out, const af_array in, const dim_type ndims, const af_index_t* indexers);

    // generalized indexing function that accepts either af_array or af_seq
    // along a dimension to index the input array and create the corresponding
    // output array
    AFAPI af_err af_assign_gen(af_array *out, const af_array lhs, const dim_type ndims, const af_index_t* indexers, const af_array rhs);

#ifdef __cplusplus
}
#endif
