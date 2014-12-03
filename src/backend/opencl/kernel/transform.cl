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

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)

Ty mul(Ty a, Tp b) { a.x = a.x * b; a.y = a.y * b; return a; }
Ty div(Ty a, Tp b) { a.x = a.x / b; a.y = a.y / b; return a; }

#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b
#define mul(a, b) ((a) * (b))
#define div(a, b) ((a) / (b))

#endif

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

void transform_n(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido)
{
    // Compute input index
    const int xidi = round(xido * tmat[0]
                         + yido * tmat[1]
                                + tmat[2]);
    const int yidi = round(xido * tmat[3]
                         + yido * tmat[4]
                                + tmat[5]);

    // Compute memory location of indices
    dim_type loci = (yidi *  in.strides[1] + xidi);
    dim_type loco = (yido * out.strides[1] + xido);

    T val; set_scalar(val, 0);
    if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = d_in[loci];

    d_out[loco] = val;
}

void transform_b(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido)
{
    dim_type loco = (yido * out.strides[1] + xido);

    // Compute input index
    const float xid = xido * tmat[0]
                    + yido * tmat[1]
                           + tmat[2];
    const float yid = xido * tmat[3]
                    + yido * tmat[4]
                           + tmat[5];

    T zero; set_scalar(zero, 0);
    if (xid < 0 || yid < 0 || in.dims[0] < xid || in.dims[1] < yid) {
        set(d_out[loco], zero);
        return;
    }

    const float grd_x = floor(xid),  grd_y = floor(yid);
    const float off_x = xid - grd_x, off_y = yid - grd_y;

    dim_type ioff = grd_y * in.strides[1] + grd_x;

    // Check if pVal and pVal + 1 are both valid indices
    bool condY = (yid < in.dims[1] - 1);
    bool condX = (xid < in.dims[0] - 1);

    // Compute weights used
    float wt00 = (1.0 - off_x) * (1.0 - off_y);
    float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
    float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
    float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

    float wt = wt00 + wt10 + wt01 + wt11;

    // Compute Weighted Values
    T v00 =                    wt00 * d_in[ioff];
    T v10 = (condY) ?          wt10 * d_in[ioff + in.strides[1]]     : zero;
    T v01 = (condX) ?          wt01 * d_in[ioff + 1]                 : zero;
    T v11 = (condX && condY) ? wt11 * d_in[ioff + in.strides[1] + 1] : zero;
    T vo = v00 + v10 + v01 + v11;

    d_out[loco] = (vo / wt);
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
    int i_idx = xx / out.dims[0];
    int t_idx = yy / out.dims[1];

    // Index in local channel -> This is output index
    int xido = xx - i_idx * out.dims[0];
    int yido = yy - t_idx * out.dims[1];

    // Global offset
    //          Offset for transform channel + Offset for image channel.
    d_out += t_idx * nimages * out.strides[2] + i_idx * out.strides[2];
    d_in  += i_idx * in.strides[2] + in.offset;

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

    INTERP(d_out, out, d_in, in, tmat, xido, yido);
}
