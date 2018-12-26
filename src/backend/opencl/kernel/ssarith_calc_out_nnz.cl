/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void csr_calc_out_nnz(global unsigned *nnzc, global int *oRowIdx, uint M,
                             global const int *lRowIdx,
                             global const int *lColIdx,
                             global const int *rRowIdx,
                             global const int *rColIdx, local uint *blkNnz) {
    const uint row = get_global_id(0);
    const uint tid = get_local_id(0);

    const bool valid = row < M;

    const uint lEnd = (valid ? lRowIdx[row + 1] : 0);
    const uint rEnd = (valid ? rRowIdx[row + 1] : 0);

    blkNnz[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint l   = (valid ? lRowIdx[row] : 0);
    uint r   = (valid ? rRowIdx[row] : 0);
    uint nnz = 0;
    while (l < lEnd && r < rEnd) {
        uint lci = lColIdx[l];
        uint rci = rColIdx[r];
        l += (lci <= rci);
        r += (lci >= rci);
        nnz++;
    }
    nnz += (lEnd - l);
    nnz += (rEnd - r);

    blkNnz[tid] = nnz;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) oRowIdx[row + 1] = nnz;

    for (uint s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) { blkNnz[tid] += blkNnz[tid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        nnz = blkNnz[0];
        atomic_add(nnzc, nnz);
    }
}
