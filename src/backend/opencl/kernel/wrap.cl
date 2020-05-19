/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void wrap(global T *optr, KParam out, global T *iptr, KParam in,
                 const int wx, const int wy, const int sx, const int sy,
                 const int px, const int py, const int nx, const int ny,
                 int groups_x, int groups_y) {
    int idx2 = get_group_id(0) / groups_x;
    int idx3 = get_group_id(1) / groups_y;

    int groupId_x = get_group_id(0) - idx2 * groups_x;
    int groupId_y = get_group_id(1) - idx3 * groups_y;

    int oidx0 = get_local_id(0) + get_local_size(0) * groupId_x;
    int oidx1 = get_local_id(1) + get_local_size(1) * groupId_y;

    optr += idx2 * out.strides[2] + idx3 * out.strides[3] + out.offset;
    iptr += idx2 * in.strides[2] + idx3 * in.strides[3] + in.offset;

    if (oidx0 >= out.dims[0] || oidx1 >= out.dims[1]) return;

    int pidx0 = oidx0 + px;
    int pidx1 = oidx1 + py;

    // The last time a value appears in the unwrapped index is padded_index /
    // stride Each previous index has the value appear "stride" locations
    // earlier We work our way back from the last index

    const int x_end = min(pidx0 / sx, nx - 1);
    const int y_end = min(pidx1 / sy, ny - 1);

    const int x_off = pidx0 - sx * x_end;
    const int y_off = pidx1 - sy * y_end;

    T val   = ZERO;
    int idx = 1;

    for (int y = y_end, yo = y_off; y >= 0 && yo < wy; yo += sy, y--) {
        int win_end_y = yo * wx;
        int dim_end_y = y * nx;

        for (int x = x_end, xo = x_off; x >= 0 && xo < wx; xo += sx, x--) {
            int win_end = win_end_y + xo;
            int dim_end = dim_end_y + x;

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
