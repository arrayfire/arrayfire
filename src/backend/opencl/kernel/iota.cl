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
                 const dim_type s0, const dim_type s1, const dim_type s2, const dim_type s3,
                 const dim_type t0, const dim_type t1, const dim_type t2, const dim_type t3,
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

    const dim_type ozw = ow * op.strides[3] + oz * op.strides[2];

    T val = (ow / t3) * s2 * s1 * s0;
    val  += (oz / t2) * s1 * s0;

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    for(dim_type oy = yy; oy < op.dims[1]; oy += incy) {
        T valY = val + (oy / t1) * s0;
        dim_type oyzw = ozw + oy * op.strides[1];
        for(dim_type ox = xx; ox < op.dims[0]; ox += incx) {
            dim_type oidx = oyzw + ox;

            out[oidx] = valY + (ox % s0);
        }
    }
}
