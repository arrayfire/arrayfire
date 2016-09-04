/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if IS_CPLX
#if USE_DOUBLE
typedef double ScalarTy;
#else
typedef float ScalarTy;
#endif
InterpInTy __mulrc(ScalarTy s, InterpInTy v)
{
    InterpInTy out = {s * v.x, s * v.y};
    return out;
}
#define MULRC(a, b) __mulrc(a, b)
#define MULCR(a, b) __mulrc(b, a)
#else
#define MULRC(a, b) (a) * (b)
#define MULCR(a, b) (a) * (b)
#endif

InterpInTy linearInterpFunc(InterpInTy val[2], InterpPosTy ratio)
{
    return MULRC((1 - ratio), val[0]) + MULRC(ratio, val[1]);
}

InterpInTy bilinearInterpFunc(InterpInTy val[2][2], InterpPosTy xratio, InterpPosTy yratio)
{
    InterpInTy res[2];
    res[0] = linearInterpFunc(val[0], xratio);
    res[1] = linearInterpFunc(val[1], xratio);
    return linearInterpFunc(res, yratio);
}

InterpInTy cubicInterpFunc(InterpInTy val[4], InterpPosTy xratio, bool spline)
{
    InterpInTy a0, a1, a2, a3;
    if (spline) {
        a0 = MULRC(-0.5, val[0]) + MULRC( 1.5, val[1]) + MULRC(-1.5, val[2]) + MULRC( 0.5, val[3]);
        a1 = MULRC( 1.0, val[0]) + MULRC(-2.5, val[1]) + MULRC( 2.0, val[2]) + MULRC(-0.5, val[3]);
        a2 = MULRC(-0.5, val[0]) + MULRC( 0.5, val[2]);
        a3 = val[1];
    } else {
        a0 = val[3] - val[2] - val[0] + val[1];
        a1 = val[0] - val[1] - a0;
        a2 = val[2] - val[0];
        a3 = val[1];
    }

    InterpPosTy xratio2 = xratio * xratio;
    InterpPosTy xratio3 = xratio2 * xratio;

    return MULCR(a0, xratio3) + MULCR(a1, xratio2) + MULCR(a2, xratio) + a3;
}

InterpInTy bicubicInterpFunc(InterpInTy val[4][4], InterpPosTy xratio, InterpPosTy yratio, bool spline)
{
    InterpInTy res[4];
    res[0] = cubicInterpFunc(val[0], xratio, spline);
    res[1] = cubicInterpFunc(val[1], xratio, spline);
    res[2] = cubicInterpFunc(val[2], xratio, spline);
    res[3] = cubicInterpFunc(val[3], xratio, spline);
    return cubicInterpFunc(res, yratio, spline);
}


#if INTERP_ORDER == 1
InterpInTy interp1(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, int method)
{
    if (method == AF_INTERP_LOWER) {
        const int idx = floor(x) + ioff;
        return d_in[idx];
    } else {
        const int idx = round(x) + ioff;
        return d_in[idx];
    }
}
#elif INTERP_ORDER == 2
InterpInTy interp1(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, int method)
{
    const int grid_x = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;    // fractional offset
    const int idx = ioff + grid_x;
    InterpInTy zero = ZERO;
    InterpInTy val[2] = {d_in[idx], x + 1 < in.dims[0] ? d_in[idx + 1] : zero};
    InterpPosTy ratio = off_x;
    if (method == AF_INTERP_LINEAR_COSINE) {
        ratio = (1 - cos(ratio * M_PI))/2;
    }
    return linearInterpFunc(val, ratio);
}
#elif INTERP_ORDER == 3
InterpInTy interp1(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, int method)
{
    const int grid_x = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;    // fractional offset
    const int idx = ioff + grid_x;

    bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0], grid_x + 2 < in.dims[0]};
    int  off[4]  = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0, cond[3] ? 2 : (cond[2] ? 1 : 0)};

    InterpInTy val[4];
    for (int i = 0; i < 4; i++) {
        val[i] = d_in[idx + off[i]];
    }
    return cubicInterpFunc(val, off_x, method == AF_INTERP_CUBIC_SPLINE);
}
#endif

#if INTERP_ORDER == 1
InterpInTy interp2(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, InterpPosTy y, int method)
{
    if (method == AF_INTERP_LOWER) {
        const int idx = ioff + floor(y) * in.strides[1] + floor(x);
        return d_in[idx];
    } else {
        const int idx = ioff + round(y) * in.strides[1] + round(x);
        return d_in[idx];
    }
}
#elif INTERP_ORDER == 2
InterpInTy interp2(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, InterpPosTy y, int method)
{
    const int grid_x = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int idx = ioff + grid_y * in.strides[1] + grid_x;

    bool condX[2] = {true, x + 1 < in.dims[0]};
    bool condY[2] = {true, y + 1 < in.dims[1]};

    InterpInTy zero = ZERO;
    InterpInTy val[2][2];
    for (int j = 0; j < 2; j++) {
        int off_y = idx + j * in.strides[1];
        for (int i = 0; i < 2; i++) {
            val[j][i] = condX[i] && condY[j] ? d_in[off_y + i] : zero;
        }
    }

    InterpPosTy xratio = off_x, yratio = off_y;
    if (method == AF_INTERP_LINEAR_COSINE) {
        xratio = (1 - cos(xratio * M_PI))/2;
        yratio = (1 - cos(yratio * M_PI))/2;
    }

    return bilinearInterpFunc(val, xratio, yratio);
}
#elif INTERP_ORDER == 3
InterpInTy interp2(__global const InterpInTy *d_in,
                   KParam in, int ioff, InterpPosTy x, InterpPosTy y, int method)
{
    const int grid_x = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int idx = ioff + grid_y * in.strides[1] + grid_x;

    //for bicubic interpolation, work with 4x4 val at a time
    InterpInTy val[4][4];

    // used for setting values at boundaries
    bool condX[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0], grid_x + 2 < in.dims[0]};
    bool condY[4] = {grid_y - 1 >= 0, true, grid_y + 1 < in.dims[1], grid_y + 2 < in.dims[1]};
    int  offX[4]  = {condX[0] ? -1 : 0, 0, condX[2] ? 1 : 0 , condX[3] ? 2 : (condX[2] ? 1 : 0)};
    int  offY[4]  = {condY[0] ? -1 : 0, 0, condY[2] ? 1 : 0 , condY[3] ? 2 : (condY[2] ? 1 : 0)};

#pragma unroll
    for (int j = 0; j < 4; j++) {
        int ioff_j = idx + offY[j] * in.strides[1];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            val[j][i] = d_in[ioff_j + offX[i]];
        }
    }

    bool spline  = method == AF_INTERP_CUBIC_SPLINE || method == AF_INTERP_BICUBIC_SPLINE;
    return bicubicInterpFunc(val, off_x, off_y, spline);
}
#endif
