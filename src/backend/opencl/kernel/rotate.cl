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
                   const tmat_t t, const dim_type nimages)
{
    // Get thread indices
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if(xx >= out.dims[0] || yy >= out.dims[1])
        return;

    INTERP(d_out, out, d_in, in, t.tmat, xx, yy, nimages);
}
