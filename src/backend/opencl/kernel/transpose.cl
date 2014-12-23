/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#if DOCONJUGATE
T doOp(T in)
{
    T out = {in.x, -in.y};
    return out;
}
#else
#define doOp(in) in
#endif

__kernel
void transpose(__global T *oData, const KParam out,
               const __global T *iData, const KParam in,
               const dim_type nonBatchBlkSize)
{
    __local T shrdMem[TILE_DIM*(TILE_DIM+1)];

    const dim_type shrdStride = TILE_DIM+1;
    // create variables to hold output dimensions
    const dim_type oDim0 = out.dims[0];
    const dim_type oDim1 = out.dims[1];
    const dim_type iDim0 = in.dims[0];
    const dim_type iDim1 = in.dims[1];

    // calculate strides
    const dim_type oStride1 = out.strides[1];
    const dim_type iStride1 = in.strides[1];

    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    // batch based block Id
    const dim_type batchId  = get_group_id(0) / nonBatchBlkSize;
    const dim_type blkIdx_x = (get_group_id(0) - batchId * nonBatchBlkSize);
    const dim_type x0 = TILE_DIM * blkIdx_x;
    const dim_type y0 = TILE_DIM * get_group_id(1);

    // calculate global indices
    dim_type gx = lx + x0;
    dim_type gy = ly + y0;

    // offset in and out based on batch id
    // also add the subBuffer offsets
    iData += batchId *  in.strides[2] + in.offset;
    oData += batchId * out.strides[2] + out.offset;

    for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        dim_type gy_ = gy + repeat;
        if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
            shrdMem[(ly + repeat) * shrdStride + lx] = iData[gy_ * iStride1 + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    gx = lx + y0;
    gy = ly + x0;

    for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        dim_type gy_ = gy + repeat;
        if (IS32MULTIPLE || (gx < oDim0 && gy_ < oDim1)) {
            oData[gy_ * oStride1 + gx] = doOp(shrdMem[lx * shrdStride + ly + repeat]);
        }
    }
}
