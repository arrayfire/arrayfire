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
    dim_type offset[4] = {0, 0, 0, 0};
    if(dim == 0) offset[0] = X.dims[0];
    if(dim == 1) offset[1] = X.dims[1];
    if(dim == 2) offset[2] = X.dims[2];
    if(dim == 3) offset[3] = X.dims[3];

    const dim_type oz = get_group_id(0) / blocksPerMatX;
    const dim_type ow = get_group_id(1) / blocksPerMatY;

    const dim_type blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const dim_type blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const dim_type xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    d_X = d_X + X.offset;
    d_Y = d_Y + Y.offset;

    if (oz < out.dims[2] && ow < out.dims[3]) {
        d_out = d_out + oz * out.strides[2] + ow * out.strides[3];
        d_X = d_X + oz * X.strides[2] + ow * X.strides[3];
        d_Y = d_Y + (oz - offset[2]) * Y.strides[2] + (ow - offset[3]) * Y.strides[3];
        bool cond2 = oz < X.dims[2] && ow < X.dims[3];

        for (dim_type oy = yy; oy < out.dims[1]; oy += incy) {
            bool cond1 = cond2 && oy < X.dims[1];
            __global Tx *d_X_ = d_X + oy * X.strides[1];
            __global Ty *d_Y_ = d_Y + (oy - offset[1]) * Y.strides[1];
            __global Tx *d_out_ = d_out + oy * out.strides[1];

            for (dim_type ox = xx; ox < out.dims[0]; ox += incx) {
                bool cond0 = cond1 && ox < X.dims[0];
                d_out_[ox] = cond0 ? d_X_[ox] : d_Y_[ox - offset[0]];
            }
        }
    }
}
