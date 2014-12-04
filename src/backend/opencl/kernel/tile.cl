/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void tile_kernel(__global T *out, __global const T *in, const KParam op, const KParam ip,
                 const dim_type blocksPerMatX, const dim_type blocksPerMatY)
{
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

    const dim_type iz = oz % ip.dims[2];
    const dim_type iw = ow % ip.dims[3];
    const dim_type izw = iw * ip.strides[3] + iz * ip.strides[2];
    const dim_type ozw = ow * op.strides[3] + oz * op.strides[2];

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    for(dim_type oy = yy; oy < op.dims[1]; oy += incy) {
        const dim_type iy = oy % ip.dims[1];
        for(dim_type ox = xx; ox < op.dims[0]; ox += incx) {
            const dim_type ix = ox % ip.dims[0];

            dim_type iMem = izw + iy * ip.strides[1] + ix;
            dim_type oMem = ozw + oy * op.strides[1] + ox;

            out[oMem] = in[ip.offset + iMem];
        }
    }
}
