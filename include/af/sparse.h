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

#ifdef __cplusplus
namespace af
{
    class array;

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_create_sparse_array(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const af_array values, const af_array rowIdx, const af_array colIdx,
                 const af_sparse_storage storage);

    AFAPI af_err af_create_sparse_array_from_ptr(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const void * const values,
                 const int * const rowIdx, const int * const colIdx,
                 const af_dtype type, const af_sparse_storage storage,
                 const af_source source);

    AFAPI af_err af_create_sparse_array_from_dense(
                 af_array *out, const af_array in,
                 const af_sparse_storage storage);

    AFAPI af_err af_sparse_convert_storage(af_array *out, const af_array in,
                                           const af_sparse_storage destStorage);

    AFAPI af_err af_sparse_get_arrays(af_array *values, af_array *rows, af_array *cols, const af_array in);

    AFAPI af_err af_sparse_get_values(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_rows(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_cols(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_num_values(dim_t *out, const af_array in);

    AFAPI af_err af_sparse_get_num_rows(dim_t *out, const af_array in);

    AFAPI af_err af_sparse_get_num_cols(dim_t *out, const af_array in);

    AFAPI af_err af_sparse_get_storage(af_sparse_storage *out, const af_array in);

#ifdef __cplusplus
}
#endif
