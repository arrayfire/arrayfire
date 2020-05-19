/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void unwrap(global T *d_out, const KParam out, global const T *d_in,
                   const KParam in, const int wx, const int wy, const int sx,
                   const int sy, const int px, const int py, const int dx,
                   const int dy, const int nx, const int reps) {
    // Compute channel and volume
    const int w = get_group_id(1) / in.dims[2];
    const int z =
        get_group_id(1) - w * in.dims[2];  // get_group_id(1) % in.dims[2];

    if (w >= in.dims[3] || z >= in.dims[2]) return;

    // Compute offset for channel and volume
    const int cOut = w * out.strides[3] + z * out.strides[2];
    const int cIn  = w * in.strides[3] + z * in.strides[2];

    // Compute the output column index
    const int id = IS_COLUMN
                       ? (get_group_id(0) * get_local_size(1) + get_local_id(1))
                       : get_global_id(0);

    if (id >= (IS_COLUMN ? out.dims[1] : out.dims[0])) return;

    // Compute the starting index of window in x and y of input
    const int startx = (id % nx) * sx;
    const int starty = (id / nx) * sy;

    const int spx = startx - px;
    const int spy = starty - py;

    // Offset the global pointers to the respective starting indices
    global T *optr       = d_out + cOut + id * (IS_COLUMN ? out.strides[1] : 1);
    global const T *iptr = d_in + cIn + in.offset;

    bool cond = (spx >= 0 && spx + (wx * dx) < in.dims[0] && spy >= 0 &&
                 spy + (wy * dy) < in.dims[1]);

    // Compute output index local to column
    int outIdx        = IS_COLUMN ? get_local_id(0) : get_local_id(1);
    const int oStride = IS_COLUMN ? get_local_size(0) : get_local_size(1);

    for (int i = 0; i < reps; i++) {
        if (outIdx >= (IS_COLUMN ? out.dims[0] : out.dims[1])) return;

        // Compute input index local to window
        const int y = outIdx / wx;
        const int x = outIdx % wx;

        const int xpad = spx + x * dx;
        const int ypad = spy + y * dy;

        // Copy
        T val = ZERO;
        if (cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 &&
                     ypad < in.dims[1])) {
            const int inIdx = ypad * in.strides[1] + xpad * in.strides[0];
            val             = iptr[inIdx];
        }

        if (IS_COLUMN) {
            optr[outIdx] = val;
        } else {
            optr[outIdx * out.strides[1]] = val;
        }

        outIdx += oStride;
    }
}
