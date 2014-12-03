/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void iota_kernel(__global T *out, const KParam op,
                 const dim_type blocksPerMatX, const dim_type blocksPerMatY)
{
    const bool mul1 = rep > 0;
    const bool mul2 = rep > 1;
    const bool mul3 = rep > 2;

    const dim_type oz = get_group_id(0) / blocksPerMatX;
    const dim_type ow = get_group_id(1) / blocksPerMatY;

    const dim_type blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const dim_type blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const dim_type xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(xx >= op.dims[0] ||
       yy >= op.dims[1] ||
       oz >= op.dims[2] ||
       ow >= op.dims[3])
        return;

    const dim_type ozw = ow * op.strides[3] + oz * op.strides[2];

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    T valOffset = mul3 * ow * op.dims[2] * op.dims[1] * op.dims[0]
                + mul2 * oz * op.dims[1] * op.dims[0];

    for(dim_type oy = yy; oy < op.dims[1]; oy += incy) {
        for(dim_type ox = xx; ox < op.dims[0]; ox += incx) {

            dim_type oidx = ozw + oy * op.strides[1] + ox;
            T val = valOffset + (mul1 * oy * op.dims[0]) + ox;

            out[oidx] = val;
        }
    }
}
