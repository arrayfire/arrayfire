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

void calc_affine_inverse(float* txo, __global const float* txi)
{
    float det = txi[0]*txi[4] - txi[1]*txi[3];

    txo[0] = txi[4] / det;
    txo[1] = txi[3] / det;
    txo[3] = txi[1] / det;
    txo[4] = txi[0] / det;

    txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
    txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
}

__kernel
void transform_kernel(__global T *d_out, const KParam out,
                      __global const T *d_in, const KParam in,
                      __global const float *c_tmat, const KParam tf,
                      const dim_type nimages, const dim_type ntransforms)
{
    // Get thread indices
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if(xx >= out.dims[0] * nimages || yy >= out.dims[1] * ntransforms)
        return;

    // Index of channel of images and transform
    //int i_idx = xx / out.dims[0];
    int t_idx = yy / out.dims[1];

    // Index in local channel -> This is output index
    int xido = xx; // - i_idx * out.dims[0];
    int yido = yy - t_idx * out.dims[1];

    // Global offset
    //          Offset for transform channel + Offset for image channel.
    d_out += t_idx * nimages * out.strides[2];// + i_idx * out.strides[2];
    //d_in  += i_idx * in.strides[2] + in.offset;

    // Transform is in global memory.
    // Needs offset to correct transform being processed.
    __global const float *tmat_ptr = c_tmat + t_idx * 6;
    float tmat[6];

    // We expect a inverse transform matrix by default
    // If it is an forward transform, then we need its inverse
    if(INVERSE == 1) {
        #pragma unroll
        for(int i = 0; i < 6; i++)
            tmat[i] = tmat_ptr[i];
    } else {
        calc_affine_inverse(tmat, tmat_ptr);
    }

    if (xido >= out.dims[0] && yido >= out.dims[1]) return;

    INTERP(d_out, out, d_in, in, tmat, xido, yido, nimages);
}
