/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/sparse.h>
#include "error.hpp"

namespace af
{
    array createSparseArray(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                            const array values, const array rowIdx, const array colIdx,
                            const af::sparseStorage storage)
    {
        af_array out = 0;
        AF_THROW(af_create_sparse_array(&out, nRows, nCols, nNZ,
                            values.get(), rowIdx.get(), colIdx.get(), storage));
        return array(out);
    }

    array createSparseArray(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                            const void * const values,
                            const int * const rowIdx, const int * const colIdx,
                            const dtype type, const af::sparseStorage storage,
                            const af::source src)
    {
        af_array out = 0;
        AF_THROW(af_create_sparse_array_from_ptr(&out, nRows, nCols, nNZ,
                            values, rowIdx, colIdx, type, storage, src));
        return array(out);
    }


    array createSparseArray(const array dense, const af::sparseStorage storage)
    {
        af_array out = 0;
        AF_THROW(af_create_sparse_array_from_dense(&out, dense.get(), storage));
        return array(out);
    }

    array sparseConvertStorage(const array in, const af::sparseStorage storage)
    {
        af_array out = 0;
        AF_THROW(af_sparse_convert_storage(&out, in.get(), storage));
        return array(out);
    }

    void sparseGetArrays(array &values, array &rowIdx, array &colIdx,
                         const array in)
    {
        af_array values_ = 0, rowIdx_ = 0, colIdx_ = 0;
        AF_THROW(af_sparse_get_arrays(&values_, &rowIdx_, &colIdx_, in.get()));
        values = array(values_);
        rowIdx = array(rowIdx_);
        colIdx = array(colIdx_);
        return;
    }

    array sparseGetValues(const array in)
    {
        af_array out = 0;
        AF_THROW(af_sparse_get_values(&out, in.get()));
        return array(out);
    }

    array sparseGetRowIdx(const array in)
    {
        af_array out = 0;
        AF_THROW(af_sparse_get_row_idx(&out, in.get()));
        return array(out);
    }

    array sparseGetColIdx(const array in)
    {
        af_array out = 0;
        AF_THROW(af_sparse_get_col_idx(&out, in.get()));
        return array(out);
    }

    dim_t sparseGetNumNonZero(const array in)
    {
        dim_t out = 0;
        AF_THROW(af_sparse_get_num_nonzero(&out, in.get()));
        return out;
    }

    af::sparseStorage sparseGetStorage(const array in)
    {
        af::sparseStorage out;
        AF_THROW(af_sparse_get_storage(&out, in.get()));
        return out;
    }
}
