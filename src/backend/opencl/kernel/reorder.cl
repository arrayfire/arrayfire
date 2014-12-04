/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void reorder_kernel(__global T *out, __global const T *in, const KParam op, const KParam ip,
                    const dim_type d0, const dim_type d1, const dim_type d2, const dim_type d3,
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

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    const dim_type o_off   = ow * op.strides[3] + oz * op.strides[2];
    const dim_type rdims[] = {d0, d1, d2, d3};
          dim_type ods[]   = {xx, yy, oz, ow};
          dim_type ids[4]  = {0};

    ids[rdims[3]] = ow;
    ids[rdims[2]] = oz;

    for(dim_type oy = yy; oy < op.dims[1]; oy += incy) {
        ids[rdims[1]] = oy;
        for(dim_type ox = xx; ox < op.dims[0]; ox += incx) {
            ids[rdims[0]] = ox;

            const dim_type oIdx = o_off + oy * op.strides[1] + ox;

            const dim_type iIdx = ids[3] * ip.strides[3] + ids[2] * ip.strides[2] +
                                  ids[1] * ip.strides[1] + ids[0];

            out[oIdx] = in[ip.offset + iIdx];
        }
    }
}
