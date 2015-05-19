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

T __cconjf(T in)
{
    T out = {in.x, -in.y};
    return out;
}

T __mul(T lhs, T rhs)
{
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

T __div(T lhs, T rhs)
{
    T out;
    TB den = (rhs.x * rhs.x + rhs.y * rhs.y);
    T num = __mul(lhs, __cconjf(rhs));

    out.x = num.x / den;
    out.y = num.y / den;

    return out;
}

#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b
#define __mul(lhs, rhs) ((lhs)*(rhs))
#define __div(lhs, rhs) ((lhs)/(rhs))

#endif

void transform_n(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido, const dim_type nimages)
{
    // Compute input index
    const dim_type xidi = round(xido * tmat[0]
                              + yido * tmat[1]
                                     + tmat[2]);
    const dim_type yidi = round(xido * tmat[3]
                              + yido * tmat[4]
                                     + tmat[5]);

    // Compute memory location of indices
    const dim_type loci = yidi * in.strides[1]  + xidi;
    const dim_type loco = yido * out.strides[1] + xido;

    for(int i = 0; i < nimages; i++) {
        // Compute memory location of indices
        dim_type ioff = loci + i * in.strides[2];
        dim_type ooff = loco + i * out.strides[2];

        T val; set_scalar(val, 0);
        if (xidi < in.dims[0] && yidi < in.dims[1] && xidi >= 0 && yidi >= 0) val = d_in[ioff];

        d_out[ooff] = val;
    }
}

void transform_b(__global T *d_out, const KParam out, __global const T *d_in, const KParam in,
                 const float *tmat, const dim_type xido, const dim_type yido, const dim_type nimages)
{
    const dim_type loco = (yido * out.strides[1] + xido);

    // Compute input index
    const float xid = xido * tmat[0]
                    + yido * tmat[1]
                           + tmat[2];
    const float yid = xido * tmat[3]
                    + yido * tmat[4]
                           + tmat[5];

    T zero; set_scalar(zero, 0);
    if (xid < -0.001 || yid < -0.001 || in.dims[0] < xid || in.dims[1] < yid) {
        for(int i = 0; i < nimages; i++) {
            set(d_out[loco + i * out.strides[2]], zero);
        }
        return;
    }

    const WT grd_x = floor(xid),  grd_y = floor(yid);
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

    const dim_type loci = grd_y * in.strides[1] + grd_x;
    for(int i = 0; i < nimages; i++) {
        const dim_type ioff = loci + (i * in.strides[2]);
        const dim_type ooff = loco + (i * out.strides[2]);

        // Compute Weighted Values
        VT v00 =                    __mul(wt00, d_in[ioff]);
        VT v10 = (condY) ?          __mul(wt10, d_in[ioff + in.strides[1]])     : zero;
        VT v01 = (condX) ?          __mul(wt01, d_in[ioff + 1])                 : zero;
        VT v11 = (condX && condY) ? __mul(wt11, d_in[ioff + in.strides[1] + 1]) : zero;
        VT vo  = v00 + v10 + v01 + v11;

        d_out[ooff] = (T)__div(vo, wt);
    }
}
