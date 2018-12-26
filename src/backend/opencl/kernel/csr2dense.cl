/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel void csr2dense(__global T *output, __global const T *values,
                        __global const int *rowidx, __global const int *colidx,
                        const int M) {
    int lid = get_local_id(0);
    for (int rowId = get_group_id(0); rowId < M; rowId += get_num_groups(0)) {
        int colStart = rowidx[rowId];
        int colEnd   = rowidx[rowId + 1];
        for (int colId = colStart + lid; colId < colEnd; colId += THREADS) {
            output[rowId + colidx[colId] * M] = values[colId];
        }
    }
}
