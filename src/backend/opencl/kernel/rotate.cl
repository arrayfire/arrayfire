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
                   const tmat_t t, const dim_type nimages, const dim_type blocksYPerImage)
{
    // Compute which image set
    const dim_type setId = get_group_id(1) / blocksYPerImage;
    const dim_type blockIdx_y = get_group_id(1) - setId * blocksYPerImage;

    // Get thread indices
    const dim_type xx = get_global_id(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    const dim_type limages = min(out.dims[2] - setId * nimages, nimages);

    if(xx >= out.dims[0] || yy >= out.dims[1])
        return;

    __global T *optr = d_out + setId * nimages * out.strides[2];
    __global const T *iptr = d_in + in.offset + setId * nimages * in.strides[2];
    INTERP(optr, out, iptr, in, t.tmat, xx, yy, limages);
}
