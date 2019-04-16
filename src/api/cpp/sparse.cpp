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

namespace af {
array sparse(const dim_t nRows, const dim_t nCols, const array values,
             const array rowIdx, const array colIdx, const af::storage stype) {
    af_array out = 0;
    AF_THROW(af_create_sparse_array(&out, nRows, nCols, values.get(),
                                    rowIdx.get(), colIdx.get(), stype));
    return array(out);
}

array sparse(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
             const void *const values, const int *const rowIdx,
             const int *const colIdx, const dtype type, const af::storage stype,
             const af::source src) {
    af_array out = 0;
    AF_THROW(af_create_sparse_array_from_ptr(&out, nRows, nCols, nNZ, values,
                                             rowIdx, colIdx, type, stype, src));
    return array(out);
}

array sparse(const array dense, const af::storage stype) {
    af_array out = 0;
    AF_THROW(af_create_sparse_array_from_dense(&out, dense.get(), stype));
    return array(out);
}

array sparseConvertTo(const array in, const af::storage stype) {
    af_array out = 0;
    AF_THROW(af_sparse_convert_to(&out, in.get(), stype));
    return array(out);
}

array dense(const array sparse) {
    af_array out = 0;
    AF_THROW(af_sparse_to_dense(&out, sparse.get()));
    return array(out);
}

void sparseGetInfo(array &values, array &rowIdx, array &colIdx, storage &stype,
                   const array in) {
    af_array values_ = 0, rowIdx_ = 0, colIdx_ = 0;
    af_storage stype_ = AF_STORAGE_DENSE;
    AF_THROW(
        af_sparse_get_info(&values_, &rowIdx_, &colIdx_, &stype_, in.get()));
    values = array(values_);
    rowIdx = array(rowIdx_);
    colIdx = array(colIdx_);
    stype  = stype_;
    return;
}

array sparseGetValues(const array in) {
    af_array out = 0;
    AF_THROW(af_sparse_get_values(&out, in.get()));
    return array(out);
}

array sparseGetRowIdx(const array in) {
    af_array out = 0;
    AF_THROW(af_sparse_get_row_idx(&out, in.get()));
    return array(out);
}

array sparseGetColIdx(const array in) {
    af_array out = 0;
    AF_THROW(af_sparse_get_col_idx(&out, in.get()));
    return array(out);
}

dim_t sparseGetNNZ(const array in) {
    dim_t out = 0;
    AF_THROW(af_sparse_get_nnz(&out, in.get()));
    return out;
}

af::storage sparseGetStorage(const array in) {
    af::storage out;
    AF_THROW(af_sparse_get_storage(&out, in.get()));
    return out;
}
}  // namespace af
