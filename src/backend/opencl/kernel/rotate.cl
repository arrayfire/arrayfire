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

typedef struct {
    float tmat[6];
} tmat_t;

kernel void rotateKernel(global T *d_out, const KParam out,
                         global const T *d_in, const KParam in,
                         const tmat_t t, const int nimages, const int batches,
                         const int blocksXPerImage, const int blocksYPerImage,
                         int method) {
    // Compute which image set
    const int setId      = get_group_id(0) / blocksXPerImage;
    const int blockIdx_x = get_group_id(0) - setId * blocksXPerImage;

    const int batch      = get_group_id(1) / blocksYPerImage;
    const int blockIdx_y = get_group_id(1) - batch * blocksYPerImage;

    // Get thread indices
    const int xido = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yido = get_local_id(1) + blockIdx_y * get_local_size(1);

    const int limages = min((int)out.dims[2] - setId * nimages, nimages);

    if (xido >= out.dims[0] || yido >= out.dims[1]) return;

    InterpPosTy xidi = xido * t.tmat[0] + yido * t.tmat[1] + t.tmat[2];
    InterpPosTy yidi = xido * t.tmat[3] + yido * t.tmat[4] + t.tmat[5];

    int outoff =
        out.offset + setId * nimages * out.strides[2] + batch * out.strides[3];
    int inoff =
        in.offset + setId * nimages * in.strides[2] + batch * in.strides[3];

    const int loco = outoff + (yido * out.strides[1] + xido);

    InterpInTy zero = ZERO;
    if (INTERP_ORDER > 1) {
        // Special conditions to deal with boundaries for bilinear and bicubic
        // FIXME: Ideally this condition should be removed or be present for all
        // methods But tests are expecting a different behavior for bilinear and
        // nearest
        if (xidi < (InterpPosTy)-0.0001 || yidi < (InterpPosTy)-0.0001 ||
            in.dims[0] <= xidi || in.dims[1] <= yidi) {
            for (int i = 0; i < nimages; i++) {
                d_out[loco + i * out.strides[2]] = zero;
            }
            return;
        }
    }

    // FIXME: Nearest and lower do not do clamping, but other methods do
    // Make it consistent
    const bool doclamp = INTERP_ORDER != 1;
    interp2(d_out, out, loco, d_in, in, inoff, xidi, yidi, method, limages,
            doclamp, 2);
}
