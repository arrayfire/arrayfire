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

void calc_transf_inverse(float *txo, global const float *txi) {
#if PERSPECTIVE
    txo[0] = txi[4] * txi[8] - txi[5] * txi[7];
    txo[1] = -(txi[1] * txi[8] - txi[2] * txi[7]);
    txo[2] = txi[1] * txi[5] - txi[2] * txi[4];

    txo[3] = -(txi[3] * txi[8] - txi[5] * txi[6]);
    txo[4] = txi[0] * txi[8] - txi[2] * txi[6];
    txo[5] = -(txi[0] * txi[5] - txi[2] * txi[3]);

    txo[6] = txi[3] * txi[7] - txi[4] * txi[6];
    txo[7] = -(txi[0] * txi[7] - txi[1] * txi[6]);
    txo[8] = txi[0] * txi[4] - txi[1] * txi[3];

    float det = txi[0] * txo[0] + txi[1] * txo[3] + txi[2] * txo[6];

    txo[0] /= det;
    txo[1] /= det;
    txo[2] /= det;
    txo[3] /= det;
    txo[4] /= det;
    txo[5] /= det;
    txo[6] /= det;
    txo[7] /= det;
    txo[8] /= det;
#else
    float det = txi[0] * txi[4] - txi[1] * txi[3];

    txo[0] = txi[4] / det;
    txo[1] = txi[3] / det;
    txo[3] = txi[1] / det;
    txo[4] = txi[0] / det;

    txo[2]               = txi[2] * -txo[0] + txi[5] * -txo[1];
    txo[5]               = txi[2] * -txo[3] + txi[5] * -txo[4];
#endif
}

kernel void transformKernel(global T *d_out, const KParam out,
                            global const T *d_in, const KParam in,
                            global const float *c_tmat, const KParam tf,
                            const int nImg2, const int nImg3, const int nTfs2,
                            const int nTfs3, const int batchImg2,
                            const int blocksXPerImage,
                            const int blocksYPerImage, const int method) {
    // Image Ids
    const int imgId2 = get_group_id(0) / blocksXPerImage;
    const int imgId3 = get_group_id(1) / blocksYPerImage;

    // Block in local image
    const int blockIdx_x = get_group_id(0) - imgId2 * blocksXPerImage;
    const int blockIdx_y = get_group_id(1) - imgId3 * blocksYPerImage;

    // Get thread indices in local image
    const int xido = blockIdx_x * get_local_size(0) + get_local_id(0);
    const int yido = blockIdx_y * get_local_size(1) + get_local_id(1);

    // Image iteration loop count for image batching
    int limages = min(max((int)(out.dims[2] - imgId2 * nImg2), 1), batchImg2);

    if (xido >= out.dims[0] || yido >= out.dims[1]) return;

    // Index of transform
    const int eTfs2 = max((nTfs2 / nImg2), 1);
    const int eTfs3 = max((nTfs3 / nImg3), 1);

    int t_idx3        = -1;  // init
    int t_idx2        = -1;  // init
    int t_idx2_offset = 0;

    const int blockIdx_z = get_group_id(2);

    if (nTfs3 == 1) {
        t_idx3 = 0;  // Always 0 as only 1 transform defined
    } else {
        if (nTfs3 == nImg3) {
            t_idx3 = imgId3;  // One to one batch with all transforms defined
        } else {
            t_idx3        = blockIdx_z / eTfs2;  // Transform batched, calculate
            t_idx2_offset = t_idx3 * nTfs2;
        }
    }

    if (nTfs2 == 1) {
        t_idx2 = 0;  // Always 0 as only 1 transform defined
    } else {
        if (nTfs2 == nImg2) {
            t_idx2 = imgId2;  // One to one batch with all transforms defined
        } else {
            t_idx2 =
                blockIdx_z - t_idx2_offset;  // Transform batched, calculate
        }
    }

    // Linear transform index
    const int t_idx = t_idx2 + t_idx3 * nTfs2;

    // Global outoff
    int outoff = out.offset;
    int inoff =
        imgId2 * batchImg2 * in.strides[2] + imgId3 * in.strides[3] + in.offset;
    if (nImg2 == nTfs2 || nImg2 > 1) {  // One-to-One or Image on dim2
        outoff += imgId2 * batchImg2 * out.strides[2];
    } else {  // Transform batched on dim2
        outoff += t_idx2 * out.strides[2];
    }

    if (nImg3 == nTfs3 || nImg3 > 1) {  // One-to-One or Image on dim3
        outoff += imgId3 * out.strides[3];
    } else {  // Transform batched on dim2
        outoff += t_idx3 * out.strides[3];
    }

    // Transform is in global memory.
    // Needs outoff to correct transform being processed.
#if PERSPECTIVE
    const int transf_len = 9;
    float tmat[9];
#else
    const int transf_len = 6;
    float tmat[6];
#endif
    global const float *tmat_ptr = c_tmat + t_idx * transf_len;

    // We expect a inverse transform matrix by default
    // If it is an forward transform, then we need its inverse
    if (INVERSE == 1) {
#pragma unroll 3
        for (int i = 0; i < transf_len; i++) tmat[i] = tmat_ptr[i];
    } else {
        calc_transf_inverse(tmat, tmat_ptr);
    }

    InterpPosTy xidi = xido * tmat[0] + yido * tmat[1] + tmat[2];
    InterpPosTy yidi = xido * tmat[3] + yido * tmat[4] + tmat[5];

#if PERSPECTIVE
    const InterpPosTy W = xido * tmat[6] + yido * tmat[7] + tmat[8];
    xidi /= W;
    yidi /= W;
#endif
    const int loco = outoff + (yido * out.strides[1] + xido);
    // FIXME: Nearest and lower do not do clamping, but other methods do
    // Make it consistent
    const bool doclamp = INTERP_ORDER != 1;

    T zero = ZERO;
    if (xidi < (InterpPosTy)-0.0001 || yidi < (InterpPosTy)-0.0001 ||
        in.dims[0] <= xidi || in.dims[1] <= yidi) {
        for (int n = 0; n < limages; n++) {
            d_out[loco + n * out.strides[2]] = zero;
        }
        return;
    }

    interp2(d_out, out, loco, d_in, in, inoff, xidi, yidi, method, limages,
            doclamp, 2);
}
