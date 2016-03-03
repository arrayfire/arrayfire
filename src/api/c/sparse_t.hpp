/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/sparse.h>

typedef struct {
    dim_t nRows, nCols, nNZ;
    af_sparse_storage storage;
    af_array rowIdx;
    af_array colIdx;
    af_array values;
} af_sparse_t;

af_sparse_array getSparseHandle(const af_sparse_t sparse);

af_sparse_t getSparse(const af_sparse_array sparseHandle);

