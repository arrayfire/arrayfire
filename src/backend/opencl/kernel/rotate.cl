/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define NEAREST transform_n
#define BILINEAR transform_b

typedef struct {
    float tmat[6];
} tmat_t;

__kernel
void rotate_kernel(__global T *d_out, const KParam out,
                   __global const T *d_in, const KParam in,
                   const tmat_t t, const int nimages, const int batches,
                   const int blocksXPerImage, const int blocksYPerImage)
{
    // Compute which image set
    const int setId = get_group_id(0) / blocksXPerImage;
    const int blockIdx_x = get_group_id(0) - setId * blocksXPerImage;

    const int batch = get_group_id(1) / blocksYPerImage;
    const int blockIdx_y = get_group_id(1) - batch * blocksYPerImage;

    // Get thread indices
    const int xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    const int limages = min((int)out.dims[2] - setId * nimages, nimages);

    if(xx >= out.dims[0] || yy >= out.dims[1])
        return;

    __global T *optr = d_out + setId * nimages * out.strides[2] + batch * out.strides[3];
    __global const T *iptr = d_in + in.offset + setId * nimages * in.strides[2] + batch * in.strides[3];
    INTERP(optr, out, iptr, in, t.tmat, xx, yy, limages);
}
