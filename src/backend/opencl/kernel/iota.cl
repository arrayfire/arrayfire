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
                 const int s0, const int s1, const int s2, const int s3,
                 const int t0, const int t1, const int t2, const int t3,
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

    const int ozw = ow * op.strides[3] + oz * op.strides[2];

    T val = (ow % s3) * s2 * s1 * s0;
    val  += (oz % s2) * s1 * s0;

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    for(int oy = yy; oy < op.dims[1]; oy += incy) {
        T valY = val + (oy % s1) * s0;
        int oyzw = ozw + oy * op.strides[1];
        for(int ox = xx; ox < op.dims[0]; ox += incx) {
            int oidx = oyzw + ox;

            out[oidx] = valY + (ox % s0);
        }
    }
}
