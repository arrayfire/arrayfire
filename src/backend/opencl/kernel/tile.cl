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
                 const int blocksPerMatX, const int blocksPerMatY)
{
    const int oz = get_group_id(0) / blocksPerMatX;
    const int ow = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const int xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(xx >= op.dims[0] ||
       yy >= op.dims[1] ||
       oz >= op.dims[2] ||
       ow >= op.dims[3])
        return;

    const int iz = oz % ip.dims[2];
    const int iw = ow % ip.dims[3];
    const int izw = iw * ip.strides[3] + iz * ip.strides[2];
    const int ozw = ow * op.strides[3] + oz * op.strides[2];

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    for(int oy = yy; oy < op.dims[1]; oy += incy) {
        const int iy = oy % ip.dims[1];
        for(int ox = xx; ox < op.dims[0]; ox += incx) {
            const int ix = ox % ip.dims[0];

            int iMem = izw + iy * ip.strides[1] + ix;
            int oMem = ozw + oy * op.strides[1] + ox;

            out[oMem] = in[ip.offset + iMem];
        }
    }
}
