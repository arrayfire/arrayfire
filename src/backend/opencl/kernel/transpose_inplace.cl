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
void transpose_inplace(__global T *iData, const KParam in,
                       const dim_type blocksPerMatX, const dim_type blocksPerMatY)
{
    __local T shrdMem_s[TILE_DIM*(TILE_DIM+1)];
    __local T shrdMem_d[TILE_DIM*(TILE_DIM+1)];

    const dim_type shrdStride = TILE_DIM+1;

    // create variables to hold output dimensions
    const dim_type iDim0 = in.dims[0];
    const dim_type iDim1 = in.dims[1];

    // calculate strides
    const dim_type iStride1 = in.strides[1];

    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    // batch based block Id
    const dim_type batchId_x  = get_group_id(0) / blocksPerMatX;
    const dim_type blockIdx_x = (get_group_id(0) - batchId_x * blocksPerMatX);

    const dim_type batchId_y  = get_group_id(1) / blocksPerMatY;
    const dim_type blockIdx_y = (get_group_id(1) - batchId_y * blocksPerMatY);

    const dim_type x0 = TILE_DIM * blockIdx_x;
    const dim_type y0 = TILE_DIM * blockIdx_y;

    __global T *iptr = iData + batchId_x * in.strides[2] + batchId_y * in.strides[3] + in.offset;

    if(blockIdx_y > blockIdx_x) {
        // calculate global indices
        dim_type gx = lx + x0;
        dim_type gy = ly + y0;
        dim_type dx = lx + y0;
        dim_type dy = ly + x0;

        // Copy to shared memory
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

            dim_type gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
                shrdMem_s[(ly + repeat) * shrdStride + lx] = iptr[gy_ * iStride1 + gx];

            dim_type dy_ = dy + repeat;
            if (IS32MULTIPLE || (dx < iDim0 && dy_< iDim1))
                shrdMem_d[(ly + repeat) * shrdStride + lx] = iptr[dy_ * iStride1 + dx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy from shared memory to global memory
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

            dim_type dy_ = dy + repeat;
            if (IS32MULTIPLE || (dx < iDim0 && dy_< iDim1))
                iptr[dy_ * iStride1 + dx] = doOp(shrdMem_s[(ly + repeat) + (shrdStride * lx)]);

            dim_type gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
                iptr[gy_ * iStride1 + gx] = doOp(shrdMem_d[(ly + repeat) + (shrdStride * lx)]);
        }

    } else if (blockIdx_y == blockIdx_x) {
        // calculate global indices
        dim_type gx = lx + x0;
        dim_type gy = ly + y0;

        // Copy to shared memory
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

            dim_type gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
                shrdMem_s[(ly + repeat) * shrdStride + lx] = iptr[gy_ * iStride1 + gx];

        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy from shared memory to global memory
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

            dim_type gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_< iDim1))
                iptr[gy_ * iStride1 + gx] = doOp(shrdMem_s[(ly + repeat) + (shrdStride * lx)]);
        }
    }
}
