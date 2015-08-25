/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void unwrap_kernel(__global T *d_out, const KParam out,
                   __global const T *d_in, const KParam in,
                   const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
                   const dim_t px, const dim_t py, const dim_t nx, const dim_t reps)
{
    // Compute channel and volume
    const dim_t w = get_group_id(1) / in.dims[2];
    const dim_t z = get_group_id(1) - w * in.dims[2]; // get_group_id(1) % in.dims[2];

    if(w >= in.dims[3] || z >= in.dims[2])
        return;

    // Compute offset for channel and volume
    const dim_t cOut = w * out.strides[3] + z * out.strides[2];
    const dim_t cIn  = w *  in.strides[3] + z *  in.strides[2];

    // Compute the output column index
    const dim_t id = is_column ?
        (get_group_id(0) * get_local_size(1) + get_local_id(1)) :
        get_global_id(0);

    if (id >= (is_column ? out.dims[1] : out.dims[0])) return;

    // Compute the starting index of window in x and y of input
    const dim_t startx = (id % nx) * sx;
    const dim_t starty = (id / nx) * sy;

    const dim_t spx = startx - px;
    const dim_t spy = starty - py;

    // Offset the global pointers to the respective starting indices
    __global       T* optr = d_out + cOut + id * (is_column ? out.strides[1] : 1);
    __global const T* iptr = d_in  + cIn + in.offset;

    bool cond = (spx >= 0 && spx + wx < in.dims[0] && spy >= 0 && spy + wy < in.dims[1]);

    for(int i = 0; i < reps; i++) {

        // Compute output index local to column
        const dim_t outIdx = is_column ?
            (i * get_local_size(0) + get_local_id(0)) :
            (i * get_local_size(1) + get_local_id(1));

        if(outIdx >= (is_column ? out.dims[0] : out.dims[1]))
            return;

        // Compute input index local to window
        const dim_t y = outIdx / wx;
        const dim_t x = outIdx % wx;

        const dim_t xpad = spx + x;
        const dim_t ypad = spy + y;

        // Copy
        T val = ZERO;
        if(cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 && ypad < in.dims[1])) {
            const dim_t inIdx = ypad * in.strides[1] + xpad * in.strides[0];
            val = iptr[inIdx];
        }

        if (is_column) {
            optr[outIdx] = val;
        } else {
            optr[outIdx * out.strides[1]] = val;
        }
    }
}
