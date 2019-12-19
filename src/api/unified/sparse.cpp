/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/sparse.h>
#include "symbol_manager.hpp"

af_err af_create_sparse_array(af_array *out, const dim_t nRows,
                              const dim_t nCols, const af_array values,
                              const af_array rowIdx, const af_array colIdx,
                              const af_storage stype) {
    CHECK_ARRAYS(values, rowIdx, colIdx);
    CALL(af_create_sparse_array, out, nRows, nCols, values, rowIdx, colIdx,
         stype);
}

af_err af_create_sparse_array_from_ptr(
    af_array *out, const dim_t nRows, const dim_t nCols, const dim_t nNZ,
    const void *const values, const int *const rowIdx, const int *const colIdx,
    const af_dtype type, const af_storage stype, const af_source source) {
    CALL(af_create_sparse_array_from_ptr, out, nRows, nCols, nNZ, values,
         rowIdx, colIdx, type, stype, source);
}

af_err af_create_sparse_array_from_dense(af_array *out, const af_array in,
                                         const af_storage stype) {
    CHECK_ARRAYS(in);
    CALL(af_create_sparse_array_from_dense, out, in, stype);
}

af_err af_sparse_convert_to(af_array *out, const af_array in,
                            const af_storage destStorage) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_convert_to, out, in, destStorage);
}

af_err af_sparse_to_dense(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_to_dense, out, in);
}

af_err af_sparse_get_info(af_array *values, af_array *rowIdx, af_array *colIdx,
                          af_storage *stype, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_info, values, rowIdx, colIdx, stype, in);
}

af_err af_sparse_get_values(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_values, out, in);
}

af_err af_sparse_get_row_idx(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_row_idx, out, in);
}

af_err af_sparse_get_col_idx(af_array *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_col_idx, out, in);
}

af_err af_sparse_get_nnz(dim_t *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_nnz, out, in);
}

af_err af_sparse_get_storage(af_storage *out, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_sparse_get_storage, out, in);
}
