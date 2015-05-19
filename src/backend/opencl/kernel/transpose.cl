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
               const int blocksPerMatX, const int blocksPerMatY)
{
    __local T shrdMem[TILE_DIM*(TILE_DIM+1)];

    const int shrdStride = TILE_DIM+1;
    // create variables to hold output dimensions
    const int oDim0 = out.dims[0];
    const int oDim1 = out.dims[1];
    const int iDim0 = in.dims[0];
    const int iDim1 = in.dims[1];

    // calculate strides
    const int oStride1 = out.strides[1];
    const int iStride1 = in.strides[1];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch based block Id
    const int batchId_x  = get_group_id(0) / blocksPerMatX;
    const int blockIdx_x = (get_group_id(0) - batchId_x * blocksPerMatX);

    const int batchId_y  = get_group_id(1) / blocksPerMatY;
    const int blockIdx_y = (get_group_id(1) - batchId_y * blocksPerMatY);

    const int x0 = TILE_DIM * blockIdx_x;
    const int y0 = TILE_DIM * blockIdx_y;

    // calculate global indices
    int gx = lx + x0;
    int gy = ly + y0;

    // offset in and out based on batch id
    // also add the subBuffer offsets
    iData += batchId_x *  in.strides[2] + batchId_y *  in.strides[3] +  in.offset;
    oData += batchId_x * out.strides[2] + batchId_y * out.strides[3] + out.offset;

    for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
            shrdMem[(ly + repeat) * shrdStride + lx] = iData[gy_ * iStride1 + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    gx = lx + y0;
    gy = ly + x0;

    for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (IS32MULTIPLE || (gx < oDim0 && gy_ < oDim1)) {
            oData[gy_ * oStride1 + gx] = doOp(shrdMem[lx * shrdStride + ly + repeat]);
        }
    }
}
