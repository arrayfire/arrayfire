/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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

void transform_n(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const int xido, const int yido, const int nimages)
{
    // Compute input index
    const int xidi = round(xido * tmat[0]
                         + yido * tmat[1]
                                + tmat[2]);
    const int yidi = round(xido * tmat[3]
                         + yido * tmat[4]
                                + tmat[5]);

    // Compute memory location of indices
    const int loci = yidi * in.strides[1]  + xidi;
    const int loco = yido * out.strides[1] + xido;

    for(int i = 0; i < nimages; i++) {
        // Compute memory location of indices
        int ioff = loci + i * in.strides[2];
        int ooff = loco + i * out.strides[2];

        T val; set_scalar(val, 0);
        if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = d_in[ioff];

        d_out[ooff] = val;
    }
}

void transform_b(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const int xido, const int yido, const int nimages)
{
    const int loco = (yido * out.strides[1] + xido);

    // Compute input index
    const float xid = xido * tmat[0]
                    + yido * tmat[1]
                           + tmat[2];
    const float yid = xido * tmat[3]
                    + yido * tmat[4]
                           + tmat[5];

    T zero = ZERO;
    if (xid < -0.001 || yid < -0.001 || in.dims[0] < xid || in.dims[1] < yid) {
        for(int i = 0; i < nimages; i++) {
            d_out[loco + i * out.strides[2]] = zero;
        }
        return;
    }

    const int grd_x = floor(xid),  grd_y = floor(yid);
    const WT off_x = xid - grd_x, off_y = yid - grd_y;

    // Check if pVal and pVal + 1 are both valid indices
    const bool condY = (yid < in.dims[1] - 1);
    const bool condX = (xid < in.dims[0] - 1);

    // Compute weights used
    const WT wt00 = (1.0 - off_x) * (1.0 - off_y);
    const WT wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
    const WT wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
    const WT wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

    const WT wt = wt00 + wt10 + wt01 + wt11;

    const int loci = grd_y * in.strides[1] + grd_x;
    for(int i = 0; i < nimages; i++) {
        const int ioff = loci + (i * in.strides[2]);
        const int ooff = loco + (i * out.strides[2]);

        // Compute Weighted Values
        VT v00 =                    (wt00 * d_in[ioff]);
        VT v10 = (condY) ?          (wt10 * d_in[ioff + in.strides[1]])     : zero;
        VT v01 = (condX) ?          (wt01 * d_in[ioff + 1])                 : zero;
        VT v11 = (condX && condY) ? (wt11 * d_in[ioff + in.strides[1] + 1]) : zero;
        VT vo  = v00 + v10 + v01 + v11;

        d_out[ooff] = (T)(vo / wt);
    }
}

void transform_l(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const int xido, const int yido, const int nimages)
{
    // Compute input index
    const int xidi = floor(xido * tmat[0]
                         + yido * tmat[1]
                                + tmat[2]);
    const int yidi = floor(xido * tmat[3]
                         + yido * tmat[4]
                                + tmat[5]);

    // Compute memory location of indices
    const int loci = yidi * in.strides[1]  + xidi;
    const int loco = yido * out.strides[1] + xido;

    for(int i = 0; i < nimages; i++) {
        // Compute memory location of indices
        int ioff = loci + i * in.strides[2];
        int ooff = loco + i * out.strides[2];

        T val; set_scalar(val, 0);
        if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = d_in[ioff];

        d_out[ooff] = val;
    }
}

