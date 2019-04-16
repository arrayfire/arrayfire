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
InterpInTy __mulrc(ScalarTy s, InterpInTy v) {
    InterpInTy out = {s * v.x, s * v.y};
    return out;
}
#define MULRC(a, b) __mulrc(a, b)
#define MULCR(a, b) __mulrc(b, a)
#else
#define MULRC(a, b) (a) * (b)
#define MULCR(a, b) (a) * (b)
#endif

InterpValTy linearInterpFunc(InterpValTy val[2], InterpPosTy ratio) {
    return MULRC((1 - ratio), val[0]) + MULRC(ratio, val[1]);
}

InterpValTy bilinearInterpFunc(InterpValTy val[2][2], InterpPosTy xratio,
                               InterpPosTy yratio) {
    InterpValTy res[2];
    res[0] = linearInterpFunc(val[0], xratio);
    res[1] = linearInterpFunc(val[1], xratio);
    return linearInterpFunc(res, yratio);
}

InterpValTy cubicInterpFunc(InterpValTy val[4], InterpPosTy xratio,
                            bool spline) {
    InterpValTy a0, a1, a2, a3;
    if (spline) {
        a0 = MULRC((InterpPosTy)-0.5, val[0]) +
             MULRC((InterpPosTy)1.5, val[1]) +
             MULRC((InterpPosTy)-1.5, val[2]) + MULRC((InterpPosTy)0.5, val[3]);

        a1 = MULRC((InterpPosTy)1.0, val[0]) +
             MULRC((InterpPosTy)-2.5, val[1]) +
             MULRC((InterpPosTy)2.0, val[2]) + MULRC((InterpPosTy)-0.5, val[3]);

        a2 = MULRC((InterpPosTy)-0.5, val[0]) + MULRC((InterpPosTy)0.5, val[2]);

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

InterpValTy bicubicInterpFunc(InterpValTy val[4][4], InterpPosTy xratio,
                              InterpPosTy yratio, bool spline) {
    InterpValTy res[4];
    res[0] = cubicInterpFunc(val[0], xratio, spline);
    res[1] = cubicInterpFunc(val[1], xratio, spline);
    res[2] = cubicInterpFunc(val[2], xratio, spline);
    res[3] = cubicInterpFunc(val[3], xratio, spline);
    return cubicInterpFunc(res, yratio, spline);
}

#if INTERP_ORDER == 1
void interp1(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, int method, int batch, bool clamp) {
    int xid         = (method == AF_INTERP_LOWER ? floor(x) : round(x));
    InterpInTy zero = ZERO;
    bool cond       = xid >= 0 && xid < in.dims[0];
    if (clamp) xid = max(0, min(xid, (int)in.dims[0]));
    const int idx = ioff + xid;
    for (int n = 0; n < batch; n++) {
        int idx_n                        = idx + n * in.strides[1];
        d_out[ooff + n * out.strides[1]] = (clamp || cond) ? d_in[idx_n] : zero;
    }
}
#elif INTERP_ORDER == 2
void interp1(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, int method, int batch, bool clamp) {
    const int grid_x        = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;  // fractional offset
    const int idx           = ioff + grid_x;
    InterpValTy zero        = ZERO;
    bool cond[2]            = {true, grid_x + 1 < in.dims[0]};
    int offx[2]             = {0, cond[1] ? 1 : 0};

    InterpPosTy ratio = off_x;
    if (method == AF_INTERP_LINEAR_COSINE) {
        ratio = (1 - cos(ratio * (InterpPosTy)M_PI)) / 2;
    }

    for (int n = 0; n < batch; n++) {
        int idx_n          = idx + n * in.strides[1];
        InterpValTy val[2] = {
            (clamp || cond[0]) ? d_in[idx_n + offx[0]] : zero,
            (clamp || cond[1]) ? d_in[idx_n + offx[1]] : zero};

        d_out[ooff + n * out.strides[1]] = linearInterpFunc(val, ratio);
    }
}
#elif INTERP_ORDER == 3
void interp1(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, int method, int batch, bool clamp) {
    const int grid_x        = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;  // fractional offset
    const int idx           = ioff + grid_x;

    bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0],
                    grid_x + 2 < in.dims[0]};
    int off[4]   = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                  cond[3] ? 2 : (cond[2] ? 1 : 0)};

    InterpValTy zero = ZERO;

    for (int n = 0; n < batch; n++) {
        InterpValTy val[4];
        int idx_n = idx + n * in.strides[1];
        for (int i = 0; i < 4; i++) {
            val[i] = (clamp || cond[i]) ? d_in[idx_n + off[i]] : zero;
        }
        bool spline                      = method == AF_INTERP_CUBIC_SPLINE;
        d_out[ooff + n * out.strides[1]] = cubicInterpFunc(val, off_x, spline);
        ;
    }
}
#endif

#if INTERP_ORDER == 1
void interp2(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, InterpPosTy y, int method, int nimages,
             bool clamp) {
    int xid = (method == AF_INTERP_LOWER ? floor(x) : round(x));
    int yid = (method == AF_INTERP_LOWER ? floor(y) : round(y));

    if (clamp) {
        xid = max(0, min(xid, (int)in.dims[0]));
        yid = max(0, min(yid, (int)in.dims[1]));
    }
    int idx = ioff + yid * in.strides[1] + xid;

    bool condX = xid >= 0 && xid < in.dims[0];
    bool condY = yid >= 0 && yid < in.dims[1];

    InterpInTy zero = ZERO;
    bool cond       = condX && condY;
    for (int n = 0; n < nimages; n++) {
        int idx_n                        = idx + n * in.strides[2];
        d_out[ooff + n * out.strides[2]] = (clamp || cond) ? d_in[idx_n] : zero;
    }
}
#elif INTERP_ORDER == 2
void interp2(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, InterpPosTy y, int method, int nimages,
             bool clamp) {
    const int grid_x        = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y        = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int idx = ioff + grid_y * in.strides[1] + grid_x;

    bool condX[2] = {true, x + 1 < in.dims[0]};
    bool condY[2] = {true, y + 1 < in.dims[1]};
    int offx[2]   = {0, condX[1] ? 1 : 0};
    int offy[2]   = {0, condY[1] ? 1 : 0};

    InterpPosTy xratio = off_x, yratio = off_y;
    if (method == AF_INTERP_LINEAR_COSINE) {
        xratio = (1 - cos(xratio * (InterpPosTy)M_PI)) / 2;
        yratio = (1 - cos(yratio * (InterpPosTy)M_PI)) / 2;
    }

    InterpValTy zero = ZERO;
    for (int n = 0; n < nimages; n++) {
        int idx_n = idx + n * in.strides[2];
        InterpValTy val[2][2];
        for (int j = 0; j < 2; j++) {
            int off_y = idx_n + offy[j] * in.strides[1];
            for (int i = 0; i < 2; i++) {
                bool cond = (clamp || (condX[i] && condY[j]));
                val[j][i] = cond ? d_in[off_y + offx[i]] : zero;
            }
        }
        d_out[ooff + n * out.strides[2]] =
            bilinearInterpFunc(val, xratio, yratio);
    }
}
#elif INTERP_ORDER == 3
void interp2(__global InterpInTy *d_out, KParam out, int ooff,
             __global const InterpInTy *d_in, KParam in, int ioff,
             InterpPosTy x, InterpPosTy y, int method, int nimages,
             bool clamp) {
    const int grid_x        = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y        = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int idx = ioff + grid_y * in.strides[1] + grid_x;

    // used for setting values at boundaries
    bool condX[4] = {grid_x - 1 >= 0, true, grid_x + 1 < in.dims[0],
                     grid_x + 2 < in.dims[0]};
    bool condY[4] = {grid_y - 1 >= 0, true, grid_y + 1 < in.dims[1],
                     grid_y + 2 < in.dims[1]};
    int offX[4]   = {condX[0] ? -1 : 0, 0, condX[2] ? 1 : 0,
                   condX[3] ? 2 : (condX[2] ? 1 : 0)};
    int offY[4]   = {condY[0] ? -1 : 0, 0, condY[2] ? 1 : 0,
                   condY[3] ? 2 : (condY[2] ? 1 : 0)};

    InterpValTy zero = ZERO;
    for (int n = 0; n < nimages; n++) {
        int idx_n = idx + n * in.strides[2];
        // for bicubic interpolation, work with 4x4 val at a time
        InterpValTy val[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int ioff_j = idx_n + offY[j] * in.strides[1];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                bool cond = (clamp || (condX[i] && condY[j]));
                val[j][i] = cond ? d_in[ioff_j + offX[i]] : zero;
            }
        }
        bool spline = method == AF_INTERP_CUBIC_SPLINE ||
                      method == AF_INTERP_BICUBIC_SPLINE;
        d_out[ooff + n * out.strides[2]] =
            bicubicInterpFunc(val, off_x, off_y, spline);
    }
}
#endif
