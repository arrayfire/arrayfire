/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void join_kernel(__global To *d_out, const KParam out,
                 __global const Ti *d_in, const KParam in,
                 const int o0, const int o1, const int o2, const int o3,
                 const int blocksPerMatX, const int blocksPerMatY)
{
    const int iz = get_group_id(0) / blocksPerMatX;
    const int iw = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - iz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - iw * blocksPerMatY;

    const int xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    d_in = d_in + in.offset;

    if (iz < in.dims[2] && iw < in.dims[3]) {
        d_out = d_out + (iz + o2) * out.strides[2] + (iw + o3) * out.strides[3];
        d_in = d_in + iz * in.strides[2] + iw * in.strides[3];

        for (int iy = yy; iy < in.dims[1]; iy += incy) {
            __global Ti *d_in_ = d_in + iy * in.strides[1];
            __global To *d_out_ = d_out + (iy + o1) * out.strides[1];

            for (int ix = xx; ix < in.dims[0]; ix += incx) {
                d_out_[ix + o0] = d_in_[ix];
            }
        }
    }
}
