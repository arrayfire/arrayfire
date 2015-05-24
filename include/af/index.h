/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/seq.h>

typedef struct af_index_t{
    // if seq is used for current dimension
    // isSeq is set to 'true' and idx.seq
    // should be used. Otherwise, idx.arr
    // should be used.
    union {
        af_array arr;
        af_seq   seq;
    } idx;
    // below variable is used to determine if
    // the current dimension is indexed using
    // af_array or af_seq
    bool     isSeq;
    bool     isBatch;
} af_index_t;


#if __cplusplus
namespace af
{

typedef af_index_t indexType;
class dim4;
class array;
class seq;

class AFAPI index {
    af_index_t impl;
    public:
    index();
    ~index();

    index(const int idx);
    index(const af::seq& s0);
    index(const af_seq& s0);
    index(const af::array& idx0);

    bool isspan() const;
    const af_index_t& get() const;
};

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
    AFAPI af_err af_index(af_array *out, const af_array in, const unsigned ndims, const af_seq* const index);


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
    AFAPI af_err af_assign_seq(af_array *out, const af_array lhs, const unsigned ndims, const af_seq* const index, const af_array rhs);

    // generalized indexing function that accepts either af_array or af_seq
    // along a dimension to index the input array and create the corresponding
    // output array
    AFAPI af_err af_index_gen(af_array *out, const af_array in, const dim_t ndims, const af_index_t* indexs);

    // generalized indexing function that accepts either af_array or af_seq
    // along a dimension to index the input array and create the corresponding
    // output array
    AFAPI af_err af_assign_gen(af_array *out, const af_array lhs, const dim_t ndims, const af_index_t* indexs, const af_array rhs);

#ifdef __cplusplus
}
#endif
