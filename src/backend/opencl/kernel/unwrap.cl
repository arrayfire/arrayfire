/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define divup(a, b) (((a)+(b)-1)/(b))

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)
#else
#define set(a, b) a = b
#define set_scalar(a, b) a = b
#endif

__kernel
void unwrap_kernel(__global T *d_out, const KParam out,
                   __global const T *d_in, const KParam in,
                   const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
                   const dim_t px, const dim_t py, const dim_t nx, const dim_t repsPerColumn)
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
    const dim_t colId = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if(colId >= out.dims[1])
        return;

    // Compute the starting index of window in x and y of input
    const dim_t startx = (colId % nx) * sx;
    const dim_t starty = (colId / nx) * sy;

    const dim_t spx = startx - px;
    const dim_t spy = starty - py;

    // Offset the global pointers to the respective starting indices
    __global       T* optr = d_out + cOut + colId * out.strides[1];
    __global const T* iptr = d_in  + cIn + in.offset;

    bool cond = false;
    if(spx >= 0 && spx + wx < in.dims[0] && spy >= 0 && spy + wy < in.dims[1])
        cond = true;


    for(int i = 0; i < repsPerColumn; i++) {
        // Compute output index local to column
        const dim_t colIndex = i * TX + get_local_id(0);

        if(colIndex >= out.dims[0])
            return;

        // Compute input index local to window
        const dim_t y = colIndex / wx;
        const dim_t x = colIndex % wx;

        const dim_t xpad = spx + x;
        const dim_t ypad = spy + y;

        const dim_t outIdx = (y * wx + x) * out.strides[0];

        // Copy
        if(cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 && ypad < in.dims[1])) {
            const dim_t inIdx = ypad * in.strides[1] + xpad * in.strides[0];
            optr[outIdx] = iptr[inIdx];
        } else {
            set_scalar(optr[outIdx], 0);
        }
    }
}
