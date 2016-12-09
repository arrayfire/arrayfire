/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void csr2coo(__global       int *orowidx,
             __global       int *ocolidx,
             __global const int *irowidx,
             __global const int *icolidx,
             const int M)
{
    int lid = get_local_id(0);
    for (int rowId = get_group_id(0); rowId < M; rowId += get_num_groups(0)) {
        int colStart = irowidx[rowId];
        int colEnd   = irowidx[rowId + 1];
        for (int colId = colStart + lid;  colId < colEnd; colId += THREADS) {
            orowidx[colId] = rowId;
            ocolidx[colId] = icolidx[colId];
        }
    }
}

__kernel
void swapIndex_kernel(__global       T   *ovalues,
                      __global       int *oindex,
                      __global const T   *ivalues,
                      __global const int *iindex,
                      __global const int *swapIdx,
                      const int nNZ)
{
    int id = get_global_id(0);
    if(id >= nNZ) return;

    int idx = swapIdx[id];

    ovalues[id] = ivalues[idx];
    oindex[id]  = iindex[idx];
}

__kernel
void csrReduce_kernel(__global       int *orowIdx,
                      __global const int *irowIdx,
                      const int M, const int nNZ)
{
    int id = get_global_id(0);

    if(id >= nNZ) return;

    int iRId  = irowIdx[id];
    int iRId1 = 0;
    if(id > 0) iRId1 = irowIdx[id - 1];

    if(id == 0) {
        orowIdx[id] = 0;
        orowIdx[M]  = nNZ;
    } else if(iRId1 != iRId) {
        for(int i = iRId1 + 1; i <= iRId; i++)
            orowIdx[i] = id;
    }

    // The last X rows are corner cases if they dont have any values
    if(id > irowIdx[nNZ - 1] && orowIdx[id] == 0) {
        orowIdx[id] = nNZ;
    }
}

