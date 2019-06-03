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

#if AF_API_VERSION >= 34
    /**
       This function converts \ref af::array of values, row indices and column
       indices into a sparse array.

       \note This function only create references of these arrays into the
             sparse data structure and does not do deep copies.

       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] values is the \ref af::array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] stype is the storage format of the sparse array
       \return \ref af::array for the sparse array

       \snippet test/sparse.cpp ex_sparse_af_arrays

       \ingroup sparse_func_create
     */
    AFAPI array sparse(const dim_t nRows, const dim_t nCols,
                       const array values, const array rowIdx, const array colIdx,
                       const af::storage stype = AF_STORAGE_CSR);
#endif

#if AF_API_VERSION >= 34
    /**
       This function converts host or device arrays of values, row indices and
       column indices into a sparse array on the device.

       \note The rules for deep copy/shallow copy/reference are the same as for
             creating a regular \ref af::array.

       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] nNZ is the number of non zero elements in the dense matrix
       \param[in] values is the host array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] type is the data type for the matrix
       \param[in] stype is the storage format of the sparse array
       \param[in] src is \ref afHost if inputs are host arrays and \ref afDevice
                  if the arrays are device arrays.
       \return \ref af::array for the sparse array

       \snippet test/sparse.cpp ex_sparse_host_arrays

       \ingroup sparse_func_create
     */
    AFAPI array sparse(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                       const void* const values,
                       const int * const rowIdx, const int * const colIdx,
                       const dtype type = f32, const af::storage stype = AF_STORAGE_CSR,
                       const af::source src = afHost);
#endif

#if AF_API_VERSION >= 34
    /**
       This function converts a dense \ref af::array into a sparse array.

       \param[in] dense is the source dense matrix
       \param[in] stype is the storage format of the sparse array
       \return \ref af::array for the sparse array with the given storage type

       \snippet test/sparse.cpp ex_sparse_from_dense

       \ingroup sparse_func_create
     */
    AFAPI array sparse(const array dense, const af::storage stype = AF_STORAGE_CSR);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the source sparse matrix to be converted
       \param[in] destStrorage is the storage format of the output sparse array
       \return \ref af::array for the sparse array with the given storage type

       \ingroup sparse_func_convert_to
     */
    AFAPI array sparseConvertTo(const array in, const af::storage destStrorage);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] sparse is the source sparse matrix
       \return dense \ref af::array from sparse

       \snippet test/sparse.cpp ex_dense_from_sparse

       \ingroup sparse_func_dense
     */
    AFAPI array dense(const array sparse);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] values stores the non-zero elements component of the sparse array
       \param[out] rowIdx stores the row indices component of the sparse array
       \param[out] colIdx stores the column indices component of the sparse array
       \param[out] stype stores the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \ingroup sparse_func_info
     */
    AFAPI void sparseGetInfo(array &values, array &rowIdx, array &colIdx, af::storage &stype,
                             const array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the input sparse matrix
       \return \ref af::array for the non-zero elements component of the sparse array

       \ingroup sparse_func_values
     */
    AFAPI array sparseGetValues(const array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the input sparse matrix
       \return \ref af::array for the row indices component of the sparse array

       \ingroup sparse_func_row_idx
     */
    AFAPI array sparseGetRowIdx(const array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the input sparse matrix
       \return \ref af::array for the column indices component of the sparse array

       \ingroup sparse_func_col_idx
     */
    AFAPI array sparseGetColIdx(const array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the input sparse matrix
       \return the number of non-zero elements of the sparse array

       \ingroup sparse_func_nnz
     */
    AFAPI dim_t sparseGetNNZ(const array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[in] in is the input sparse matrix
       \return \ref af::storage for the storage type of the sparse array

       \ingroup sparse_func_storage
     */
    AFAPI af::storage sparseGetStorage(const array in);
#endif
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 34
    /**
       This function converts \ref af::array of values, row indices and column
       indices into a sparse array.

       \note This function only create references of these arrays into the
             sparse data structure and does not do deep copies.

       \param[out] out \ref af::array for the sparse array
       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] values is the \ref af_array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] stype is the storage format of the sparse array

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    AFAPI af_err af_create_sparse_array(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols,
                 const af_array values, const af_array rowIdx, const af_array colIdx,
                 const af_storage stype);
#endif

#if AF_API_VERSION >= 34
    /**
       This function converts host or device arrays of values, row indices and
       column indices into a sparse array on the device.

       \note The rules for deep copy/shallow copy/reference are the same as for
             creating a regular \ref af::array.

       \param[out] out \ref af::array for the sparse array
       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] nNZ is the number of non zero elements in the dense matrix
       \param[in] values is the host array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] type is the data type for the matrix
       \param[in] stype is the storage format of the sparse array
       \param[in] src is \ref afHost if inputs are host arrays and \ref afDevice
                  if the arrays are device arrays.

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    AFAPI af_err af_create_sparse_array_from_ptr(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const void * const values,
                 const int * const rowIdx, const int * const colIdx,
                 const af_dtype type, const af_storage stype,
                 const af_source src);
#endif

#if AF_API_VERSION >= 34
    /**
       This function converts a dense \ref af_array into a sparse array.

       \param[out] out \ref af_array for the sparse array with the given storage type
       \param[in] dense is the source dense matrix
       \param[in] stype is the storage format of the sparse array

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    AFAPI af_err af_create_sparse_array_from_dense(
                 af_array *out, const af_array dense,
                 const af_storage stype);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out \ref af_array for the sparse array with the given storage type
       \param[in] in is the source sparse matrix to be converted
       \param[in] destStorage is the storage format of the output sparse array

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_convert_to
     */
    AFAPI af_err af_sparse_convert_to(af_array *out, const af_array in,
                                      const af_storage destStorage);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out dense \ref af_array from sparse
       \param[in] sparse is the source sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_dense
     */
    AFAPI af_err af_sparse_to_dense(af_array *out, const af_array sparse);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] values stores the non-zero elements component of the sparse array
       \param[out] rowIdx stores the row indices component of the sparse array
       \param[out] colIdx stores the column indices component of the sparse array
       \param[out] stype stores the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_info
     */
    AFAPI af_err af_sparse_get_info(af_array *values, af_array *rowIdx, af_array *colIdx, af_storage *stype,
                                    const af_array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out \ref af_array for the non-zero elements component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_values
     */
    AFAPI af_err af_sparse_get_values(af_array *out, const af_array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out \ref af_array for the row indices component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_row_idx
     */
    AFAPI af_err af_sparse_get_row_idx(af_array *out, const af_array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out \ref af_array for the column indices component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_col_idx
     */
    AFAPI af_err af_sparse_get_col_idx(af_array *out, const af_array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out the number of non-zero elements of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_nnz
     */
    AFAPI af_err af_sparse_get_nnz(dim_t *out, const af_array in);
#endif

#if AF_API_VERSION >= 34
    /**
       \param[out] out contains \ref af_storage for the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sparse_func_storage
     */
    AFAPI af_err af_sparse_get_storage(af_storage *out, const af_array in);
#endif

#ifdef __cplusplus
}
#endif
