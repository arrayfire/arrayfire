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

    AFAPI array createSparseArray(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                                  const array values, const array rowIdx, const array colIdx,
                                  const af::sparseStorage storage = AF_SPARSE_CSR);

    AFAPI array createSparseArray(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                                  const void* const values,
                                  const int * const rowIdx, const int * const colIdx,
                                  const dtype type = f32, const af::sparseStorage storage = AF_SPARSE_CSR,
                                  const af::source src = afHost);

    AFAPI array createSparseArray(const array dense, const af::sparseStorage storage = AF_SPARSE_CSR);

    AFAPI array sparseConvertStorage(const array in, const af::sparseStorage storage);

    AFAPI void sparseGetArrays(array &values, array &rowIdx, array &colIdx, const array in);

    AFAPI array sparseGetValues(const array in);

    AFAPI array sparseGetRowIdx(const array in);

    AFAPI array sparseGetColIdx(const array in);

    AFAPI dim_t sparseGetNumNonZero(const array in);

    AFAPI af::sparseStorage sparseGetStorage(const array in);
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

    AFAPI af_err af_sparse_get_arrays(af_array *values, af_array *rowIdx, af_array *colIdx, const af_array in);

    AFAPI af_err af_sparse_get_values(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_row_idx(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_col_idx(af_array *out, const af_array in);

    AFAPI af_err af_sparse_get_num_nonzero(dim_t *out, const af_array in);

    AFAPI af_err af_sparse_get_storage(af_sparse_storage *out, const af_array in);

#ifdef __cplusplus
}
#endif
