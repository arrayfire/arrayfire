/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void join_kernel(__global Tx *d_out, const KParam out,
                 __global const Tx *d_X, const KParam X,
                 __global const Ty *d_Y, const KParam Y,
                 const dim_type blocksPerMatX, const dim_type blocksPerMatY)
{
    dim_type offset[4];
    offset[0] = (dim == 0) ? X.dims[0] : 0;
    offset[1] = (dim == 1) ? X.dims[1] : 0;
    offset[2] = (dim == 2) ? X.dims[2] : 0;
    offset[3] = (dim == 3) ? X.dims[3] : 0;

    const dim_type oz = get_group_id(0) / blocksPerMatX;
    const dim_type ow = get_group_id(1) / blocksPerMatY;

    const dim_type blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const dim_type blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const dim_type xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(xx >= out.dims[0] ||
       yy >= out.dims[1] ||
       oz >= out.dims[2] ||
       ow >= out.dims[3])
        return;

    dim_type odx[] = {xx, yy, oz, ow};
    dim_type idx[] = {xx, yy, oz, ow};

    // These if(dim == <dimensions>) conditions are used to check which array
    // (X or Y) to use. 3 out of the 4 if conditions will not be executed
    // since the kernel is templated.
    // These if-conds decide whether to use X or Y based on the output index
    // They also compute the corrent input index if Y is chosen
    __global Tx const *in = d_X;
    dim_type const *str = X.strides;
    dim_type p_off = X.offset;
    if(dim == 2) {
        if(odx[2] >= X.dims[2]) {
            in = d_Y;
            str = Y.strides;
            p_off = Y.offset;
            idx[2] = odx[2] - offset[2];
        }
    } else if (dim == 3) {
        if(odx[3] >= X.dims[3]) {
            in = d_Y;
            str = Y.strides;
            p_off = Y.offset;
            idx[3] = odx[3] - offset[3];
        }
    }

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    const dim_type ozw = odx[3] * out.strides[3] + odx[2] * out.strides[2];

    for(dim_type oy = yy; oy < out.dims[1]; oy += incy) {
        odx[1] = oy;
        idx[1] = oy;
        if(dim == 1) {
            in = d_X;
            str = X.strides;
            p_off = X.offset;
            if(odx[1] >= X.dims[1]) {
                in = d_Y;
                str = Y.strides;
                p_off = Y.offset;
                idx[1] = odx[1] - offset[1];
            }
        }

        for(dim_type ox = xx; ox < out.dims[0]; ox += incx) {
            odx[0] = ox;
            idx[0] = ox;
            if(dim == 0) {
                in = d_X;
                str = X.strides;
                p_off = X.offset;
                if(odx[0] >= X.dims[0]) {
                    in = d_Y;
                    str = Y.strides;
                    p_off = Y.offset;
                    idx[0] = odx[0] - offset[0];
                }
            }

            const dim_type izw = idx[3] * str[3] + idx[2] * str[2];
            dim_type iMem = izw + idx[1] * str[1] + idx[0];
            dim_type oMem = ozw + odx[1] * out.strides[1] + odx[0];

            d_out[oMem] = in[p_off + iMem];
        }
    }
}
