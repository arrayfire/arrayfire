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

af_err af_create_sparse_array(
       af_array *out,
       const dim_t nRows, const dim_t nCols, const dim_t nNZ,
       const af_array values, const af_array rowIdx, const af_array colIdx,
       const af_sparse_storage storage)
{
    CHECK_ARRAYS(values, rowIdx, colIdx);
    return CALL(out, nRows, nCols, nNZ, values, rowIdx, colIdx, storage);
}

af_err af_create_sparse_array_from_ptr(
       af_array *out,
       const dim_t nRows, const dim_t nCols, const dim_t nNZ,
       const void * const values,
       const int * const rowIdx, const int * const colIdx,
       const af_dtype type, const af_sparse_storage storage,
       const af_source source)
{
    return CALL(out, nRows, nCols, nNZ, values, rowIdx, colIdx, type, storage, source);
}

af_err af_create_sparse_array_from_dense(
       af_array *out, const af_array in,
       const af_sparse_storage storage)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, storage);
}

af_err af_sparse_convert_storage(af_array *out, const af_array in,
                                 const af_sparse_storage destStorage)
{
    CHECK_ARRAYS(in);
    return CALL(out, in, destStorage);
}

af_err af_sparse_get_arrays(af_array *values, af_array *rowIdx, af_array *colIdx, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(values, rowIdx, colIdx, in);
}

af_err af_sparse_get_values(af_array *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_sparse_get_row_idx(af_array *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_sparse_get_col_idx(af_array *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_sparse_get_num_nonzero(dim_t *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}

af_err af_sparse_get_storage(af_sparse_storage *out, const af_array in)
{
    CHECK_ARRAYS(in);
    return CALL(out, in);
}
