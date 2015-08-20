/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void wrap_kernel(__global T *optr, KParam out,
                 __global T *iptr, KParam in,
                 const dim_t wx, const dim_t wy,
                 const dim_t sx, const dim_t sy,
                 const dim_t px, const dim_t py,
                 const dim_t nx, const dim_t ny,
                 dim_t groups_x,
                 dim_t groups_y)
{
    dim_t idx2 = get_group_id(0) / groups_x;
    dim_t idx3 = get_group_id(1) / groups_y;

    dim_t groupId_x = get_group_id(0) - idx2 * groups_x;
    dim_t groupId_y = get_group_id(1) - idx3 * groups_y;

    dim_t oidx0 = get_local_id(0) + get_local_size(0) * groupId_x;
    dim_t oidx1 = get_local_id(1) + get_local_size(1) * groupId_y;

    optr += idx2 * out.strides[2] + idx3 * out.strides[3];
    iptr += idx2 *  in.strides[2] + idx3 *  in.strides[3];


    if (oidx0 >= out.dims[0] || oidx1 >= out.dims[1]) return;

    dim_t pidx0 = oidx0 + px;
    dim_t pidx1 = oidx1 + py;

    // The last time a value appears in the unwrapped index is padded_index / stride
    // Each previous index has the value appear "stride" locations earlier
    // We work our way back from the last index

    const dim_t x_end = min(pidx0 / sx, nx - 1);
    const dim_t y_end = min(pidx1 / sy, ny - 1);

    const dim_t x_off = pidx0 - sx * x_end;
    const dim_t y_off = pidx1 - sy * y_end;

    T val = ZERO;
    dim_t idx = 1;

    for (dim_t y = y_end, yo = y_off; y >= 0 && yo < wy; yo += sy, y--) {
        dim_t win_end_y = yo * wx;
        dim_t dim_end_y = y * nx;

        for (dim_t x = x_end, xo = x_off; x >= 0 && xo < wx; xo += sx, x--) {

            dim_t win_end = win_end_y + xo;
            dim_t dim_end = dim_end_y + x;

            if (is_column) {
                idx = dim_end * in.strides[1] + win_end;
            } else {
                idx = dim_end + win_end * in.strides[1];
            }

            // No need to include anything special for complex
            // Add for complex numbers is just vector add of reals
            // Might need to change if we generalize add to more binary ops
            val = val + iptr[idx];
        }
    }

    optr[oidx1 * out.strides[1] + oidx0] = val;
}
