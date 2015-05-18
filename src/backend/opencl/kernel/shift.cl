/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

static inline int simple_mod(const int i, const int dim)
{
    return (i < dim) ? i : (i - dim);
}

__kernel
void shift_kernel(__global T *out, __global const T *in, const KParam op, const KParam ip,
                  const int d0, const int d1, const int d2, const int d3,
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

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    const int iw = simple_mod((ow + d3), op.dims[3]);
    const int iz = simple_mod((oz + d2), op.dims[2]);

    const int o_off   = ow * op.strides[3] + oz * op.strides[2];
    const int i_off   = iw * ip.strides[3] + iz * ip.strides[2] + ip.offset;

    for(int oy = yy; oy < op.dims[1]; oy += incy) {
        const int iy = simple_mod((oy + d1), op.dims[1]);
        for(int ox = xx; ox < op.dims[0]; ox += incx) {
            const int ix = simple_mod((ox + d0), op.dims[0]);

            const int oIdx = o_off + oy * op.strides[1] + ox;
            const int iIdx = i_off + iy * ip.strides[1] + ix;

            out[oIdx] = in[iIdx];
        }
    }
}
