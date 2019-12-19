/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/index.h>
#include "symbol_manager.hpp"

af_err af_index(af_array* out, const af_array in, const unsigned ndims,
                const af_seq* const index) {
    CHECK_ARRAYS(in);
    CALL(af_index, out, in, ndims, index);
}

af_err af_lookup(af_array* out, const af_array in, const af_array indices,
                 const unsigned dim) {
    CHECK_ARRAYS(in, indices);
    CALL(af_lookup, out, in, indices, dim);
}

af_err af_assign_seq(af_array* out, const af_array lhs, const unsigned ndims,
                     const af_seq* const indices, const af_array rhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(af_assign_seq, out, lhs, ndims, indices, rhs);
}

af_err af_index_gen(af_array* out, const af_array in, const dim_t ndims,
                    const af_index_t* indices) {
    CHECK_ARRAYS(in);
    CALL(af_index_gen, out, in, ndims, indices);
}

af_err af_assign_gen(af_array* out, const af_array lhs, const dim_t ndims,
                     const af_index_t* indices, const af_array rhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(af_assign_gen, out, lhs, ndims, indices, rhs);
}

af_seq af_make_seq(double begin, double end, double step) {
    af_seq seq = {begin, end, step};
    return seq;
}

af_err af_create_indexers(af_index_t** indexers) {
    CALL(af_create_indexers, indexers);
}

af_err af_set_array_indexer(af_index_t* indexer, const af_array idx,
                            const dim_t dim) {
    CHECK_ARRAYS(idx);
    CALL(af_set_array_indexer, indexer, idx, dim);
}

af_err af_set_seq_indexer(af_index_t* indexer, const af_seq* idx,
                          const dim_t dim, const bool is_batch) {
    CALL(af_set_seq_indexer, indexer, idx, dim, is_batch);
}

af_err af_set_seq_param_indexer(af_index_t* indexer, const double begin,
                                const double end, const double step,
                                const dim_t dim, const bool is_batch) {
    CALL(af_set_seq_param_indexer, indexer, begin, end, step, dim, is_batch);
}

af_err af_release_indexers(af_index_t* indexers) {
    CALL(af_release_indexers, indexers);
}
