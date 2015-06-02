/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void range_kernel(__global T *out, const KParam op, const int dim,
                 const int blocksPerMatX, const int blocksPerMatY)
{
    const int mul0 = (dim == 0);
    const int mul1 = (dim == 1);
    const int mul2 = (dim == 2);
    const int mul3 = (dim == 3);

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

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    T valZW = (mul3 * ow) + (mul2 * oz);

    for(int oy = yy; oy < op.dims[1]; oy += incy) {
        T valYZW = valZW + (mul1 * oy);
        int oyzw = ozw + oy * op.strides[1];
        for(int ox = xx; ox < op.dims[0]; ox += incx) {
            int oidx = oyzw + ox;
            T val = valYZW + (mul0 * ox);

            out[oidx] = val;
        }
    }
}
