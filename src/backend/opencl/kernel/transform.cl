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
#define LOWER transform_l

void calc_transf_inverse(float* txo, __global const float* txi)
{
#if PERSPECTIVE
    txo[0] =   txi[4]*txi[8] - txi[5]*txi[7];
    txo[1] = -(txi[1]*txi[8] - txi[2]*txi[7]);
    txo[2] =   txi[1]*txi[5] - txi[2]*txi[4];

    txo[3] = -(txi[3]*txi[8] - txi[5]*txi[6]);
    txo[4] =   txi[0]*txi[8] - txi[2]*txi[6];
    txo[5] = -(txi[0]*txi[5] - txi[2]*txi[3]);

    txo[6] =   txi[3]*txi[7] - txi[4]*txi[6];
    txo[7] = -(txi[0]*txi[7] - txi[1]*txi[6]);
    txo[8] =   txi[0]*txi[4] - txi[1]*txi[3];

    float det = txi[0]*txo[0] + txi[1]*txo[3] + txi[2]*txo[6];

    txo[0] /= det; txo[1] /= det; txo[2] /= det;
    txo[3] /= det; txo[4] /= det; txo[5] /= det;
    txo[6] /= det; txo[7] /= det; txo[8] /= det;
#else
    float det = txi[0]*txi[4] - txi[1]*txi[3];

    txo[0] = txi[4] / det;
    txo[1] = txi[3] / det;
    txo[3] = txi[1] / det;
    txo[4] = txi[0] / det;

    txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
    txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
#endif
}

__kernel
void transform_kernel(__global T *d_out, const KParam out,
                      __global const T *d_in, const KParam in,
                      __global const float *c_tmat, const KParam tf,
                      const int nimages, const int ntransforms,
                      const int blocksXPerImage)
{
    // Compute which image set
    const int setId = get_group_id(0) / blocksXPerImage;
    const int blockIdx_x = get_group_id(0) - setId * blocksXPerImage;

    // Get thread indices
    const int xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yy = get_global_id(1);

    if(xx >= out.dims[0] * nimages || yy >= out.dims[1] * ntransforms)
        return;

    // Index of channel of images and transform
    //int i_idx = xx / out.dims[0];
    const int t_idx = yy / out.dims[1];

    const int limages = min((int)out.dims[2] - setId * nimages, nimages);

    // Index in local channel -> This is output index
    const int xido = xx; // - i_idx * out.dims[0];
    const int yido = yy - t_idx * out.dims[1];

    // Global offset
    //          Offset for transform channel + Offset for image channel.
    d_out += t_idx * nimages * out.strides[2] + setId * nimages * out.strides[2];
    d_in  += setId * nimages * in.strides[2] + in.offset;

    // Transform is in global memory.
    // Needs offset to correct transform being processed.
    __global const float *tmat_ptr = c_tmat + t_idx * TRANSF_LEN;
    float tmat[TRANSF_LEN];

    // We expect a inverse transform matrix by default
    // If it is an forward transform, then we need its inverse
    if(INVERSE == 1) {
        #pragma unroll 3
        for(int i = 0; i < TRANSF_LEN; i++)
            tmat[i] = tmat_ptr[i];
    } else {
        calc_transf_inverse(tmat, tmat_ptr);
    }

    if (xido >= out.dims[0] && yido >= out.dims[1]) return;

    INTERP(d_out, out, d_in, in, tmat, xido, yido, limages);
}
