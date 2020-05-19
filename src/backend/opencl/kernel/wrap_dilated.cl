/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void wrap_dilated(global T *optr, KParam out, global T *iptr, KParam in,
                         const int wx, const int wy, const int sx, const int sy,
                         const int px, const int py, const int dx, const int dy,
                         const int nx, const int ny, int groups_x,
                         int groups_y) {
    int idx2 = get_group_id(0) / groups_x;
    int idx3 = get_group_id(1) / groups_y;

    int groupId_x = get_group_id(0) - idx2 * groups_x;
    int groupId_y = get_group_id(1) - idx3 * groups_y;

    int oidx0 = get_local_id(0) + get_local_size(0) * groupId_x;
    int oidx1 = get_local_id(1) + get_local_size(1) * groupId_y;

    optr += idx2 * out.strides[2] + idx3 * out.strides[3];
    iptr += idx2 * in.strides[2] + idx3 * in.strides[3] + in.offset;

    if (oidx0 >= out.dims[0] || oidx1 >= out.dims[1]) return;

    int eff_wx = wx + (wx - 1) * (dx - 1);
    int eff_wy = wy + (wy - 1) * (dy - 1);

    int pidx0 = oidx0 + px;
    int pidx1 = oidx1 + py;

    // The last time a value appears in the unwrapped index is padded_index /
    // stride
    // Each previous index has the value appear "stride" locations earlier
    // We work our way back from the last index

    const int y_start = (pidx1 < eff_wy) ? 0 : (pidx1 - eff_wy) / sy + 1;
    const int y_end   = min(pidx1 / sy + 1, ny);

    const int x_start = (pidx0 < eff_wx) ? 0 : (pidx0 - eff_wx) / sx + 1;
    const int x_end   = min(pidx0 / sx + 1, nx);

    T val   = ZERO;
    int idx = 1;

    for (int y = y_start; y < y_end; y++) {
        int fy      = (pidx1 - y * sy);
        bool yvalid = (fy % dy == 0) && (y < ny);
        fy /= dy;

        int win_end_y = fy * wx;
        int dim_end_y = y * nx;

        for (int x = x_start; x < x_end; x++) {
            int fx      = (pidx0 - x * sx);
            bool xvalid = (fx % dx == 0) && (x < nx);
            fx /= dx;

            int win_end = win_end_y + fx;
            int dim_end = dim_end_y + x;

            if (is_column) {
                idx = dim_end * in.strides[1] + win_end;
            } else {
                idx = dim_end + win_end * in.strides[1];
            }

            T ival;
            ival = (yvalid && xvalid) ? iptr[idx] : ZERO;
            val  = val + ival;
        }
    }

    optr[oidx1 * out.strides[1] + oidx0] = val;
}
