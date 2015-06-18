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
                   const dim_t repsPerColumn)
{
    // Compute channel and volume
    const dim_t w = get_group_id(1) / in.dims[2];
    const dim_t z = get_group_id(1) - w * in.dims[2]; // get_group_id(1) % in.dims[2];

    if(w >= in.dims[3] || z >= in.dims[2])
        return;

    // Compute offset for channel and volume
    const dim_t cOut = w * out.strides[3] + z * out.strides[2];
    const dim_t cIn  = w *  in.strides[3] + z *  in.strides[2];

    // Compute the number of windows along dim0 of input
    const dim_t nx = (in.dims[0] - wx) / sx + 1;
    //dim_t ny = (in.dims[1] - wy) / sy + 1;

    // Compute the output column index
    const dim_t colId = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if(colId >= out.dims[1])
        return;

    // Compute the starting index of window in x and y of input
    const dim_t startx = (colId % nx) * sx;
    const dim_t starty = (colId / nx) * sy;

    // Offset the global pointers to the respective starting indices
    __global T* optr = d_out + cOut + colId * out.strides[1];
    __global T* iptr = d_in  + cIn  + starty * in.strides[1] + startx + in.offset;

    for(int i = 0; i < repsPerColumn; i++) {
        // Compute output index local to column
        const dim_t colIndex = i * TX + get_local_id(0);

        if(colIndex >= out.dims[0])
            return;

        // Compute input index local to window
        const dim_t y = colIndex / wx;
        const dim_t x = colIndex - y * wx;  // colIndex % wx

        const dim_t outIdx = (y * wx + x) * out.strides[0];
        const dim_t inIdx = y * in.strides[1] + x * in.strides[0];

        // Copy
        optr[outIdx] = iptr[inIdx];
    }
}
