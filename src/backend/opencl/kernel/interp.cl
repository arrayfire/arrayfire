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
void interp1(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             int method, int batch, bool doclamp, int batch_dim) {
    InterpInTy zero = ZERO;

    const int x_lim    = in.dims[XDIM];
    const int x_stride = in.strides[XDIM];

    int xid   = (method == AF_INTERP_LOWER ? floor(x) : round(x));
    bool cond = xid >= 0 && xid < x_lim;
    if (doclamp) xid = max(0, min(xid, x_lim));

    const int idx = ioff + xid * x_stride;

    for (int n = 0; n < batch; n++) {
        int idx_n = idx + n * in.strides[batch_dim];
        d_out[ooff + n * out.strides[batch_dim]] =
            (doclamp || cond) ? d_in[idx_n] : zero;
    }
}
#elif INTERP_ORDER == 2
void interp1(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             int method, int batch, bool doclamp, int batch_dim) {
    const int grid_x        = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;  // fractional offset

    const int x_lim    = in.dims[XDIM];
    const int x_stride = in.strides[XDIM];
    const int idx      = ioff + grid_x * x_stride;

    InterpValTy zero  = ZERO;
    bool cond[2]      = {true, grid_x + 1 < x_lim};
    int offx[2]       = {0, cond[1] ? 1 : 0};
    InterpPosTy ratio = off_x;
    if (method == AF_INTERP_LINEAR_COSINE) {
        ratio = (1 - cos(ratio * (InterpPosTy)M_PI)) / 2;
    }

    for (int n = 0; n < batch; n++) {
        int idx_n          = idx + n * in.strides[batch_dim];
        InterpValTy val[2] = {
            (doclamp || cond[0]) ? d_in[idx_n + offx[0] * x_stride] : zero,
            (doclamp || cond[1]) ? d_in[idx_n + offx[1] * x_stride] : zero};

        d_out[ooff + n * out.strides[batch_dim]] = linearInterpFunc(val, ratio);
    }
}
#elif INTERP_ORDER == 3
void interp1(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             int method, int batch, bool doclamp, int batch_dim) {
    const int grid_x        = floor(x);    // nearest grid
    const InterpPosTy off_x = x - grid_x;  // fractional offset

    const int x_lim    = in.dims[XDIM];
    const int x_stride = in.strides[XDIM];
    const int idx      = ioff + grid_x * x_stride;

    bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                    grid_x + 2 < x_lim};
    int off[4]   = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                  cond[3] ? 2 : (cond[2] ? 1 : 0)};

    InterpValTy zero = ZERO;

    for (int n = 0; n < batch; n++) {
        InterpValTy val[4];
        int idx_n = idx + n * in.strides[batch_dim];
        for (int i = 0; i < 4; i++) {
            val[i] =
                (doclamp || cond[i]) ? d_in[idx_n + off[i] * x_stride] : zero;
        }
        bool spline = method == AF_INTERP_CUBIC_SPLINE;
        d_out[ooff + n * out.strides[batch_dim]] =
            cubicInterpFunc(val, off_x, spline);
        ;
    }
}
#endif

#if defined(YDIM)  // If 2d interpolation is being used
#if INTERP_ORDER == 1
void interp2(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             InterpPosTy y, int method, int batch, bool doclamp,
             int batch_dim) {
    int xid = (method == AF_INTERP_LOWER ? floor(x) : round(x));
    int yid = (method == AF_INTERP_LOWER ? floor(y) : round(y));

    const int x_lim    = in.dims[XDIM];
    const int y_lim    = in.dims[YDIM];
    const int x_stride = in.strides[XDIM];
    const int y_stride = in.strides[YDIM];

    if (doclamp) {
        xid = max(0, min(xid, x_lim));
        yid = max(0, min(yid, y_lim));
    }
    const int idx = ioff + yid * y_stride + xid * x_stride;

    bool condX = xid >= 0 && xid < x_lim;
    bool condY = yid >= 0 && yid < y_lim;

    InterpInTy zero = ZERO;
    bool cond       = condX && condY;
    for (int n = 0; n < batch; n++) {
        int idx_n = idx + n * in.strides[batch_dim];
        d_out[ooff + n * out.strides[batch_dim]] =
            (doclamp || cond) ? d_in[idx_n] : zero;
    }
}
#elif INTERP_ORDER == 2
void interp2(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             InterpPosTy y, int method, int batch, bool doclamp,
             int batch_dim) {
    const int grid_x        = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y        = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int x_lim    = in.dims[XDIM];
    const int y_lim    = in.dims[YDIM];
    const int x_stride = in.strides[XDIM];
    const int y_stride = in.strides[YDIM];
    const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

    bool condX[2] = {true, x + 1 < x_lim};
    bool condY[2] = {true, y + 1 < y_lim};
    int offx[2]   = {0, condX[1] ? 1 : 0};
    int offy[2]   = {0, condY[1] ? 1 : 0};

    InterpPosTy xratio = off_x, yratio = off_y;
    if (method == AF_INTERP_LINEAR_COSINE) {
        xratio = (1 - cos(xratio * (InterpPosTy)M_PI)) / 2;
        yratio = (1 - cos(yratio * (InterpPosTy)M_PI)) / 2;
    }

    InterpValTy zero = ZERO;
    for (int n = 0; n < batch; n++) {
        int idx_n = idx + n * in.strides[batch_dim];
        InterpValTy val[2][2];
        for (int j = 0; j < 2; j++) {
            int off_y = idx_n + offy[j] * y_stride;
            for (int i = 0; i < 2; i++) {
                bool cond = (doclamp || (condX[i] && condY[j]));
                val[j][i] = cond ? d_in[off_y + offx[i] * x_stride] : zero;
            }
        }
        d_out[ooff + n * out.strides[batch_dim]] =
            bilinearInterpFunc(val, xratio, yratio);
    }
}
#elif INTERP_ORDER == 3
void interp2(global InterpInTy *d_out, KParam out, int ooff,
             global const InterpInTy *d_in, KParam in, int ioff, InterpPosTy x,
             InterpPosTy y, int method, int batch, bool doclamp,
             int batch_dim) {
    const int grid_x        = floor(x);
    const InterpPosTy off_x = x - grid_x;

    const int grid_y        = floor(y);
    const InterpPosTy off_y = y - grid_y;

    const int x_lim    = in.dims[XDIM];
    const int y_lim    = in.dims[YDIM];
    const int x_stride = in.strides[XDIM];
    const int y_stride = in.strides[YDIM];
    const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

    // used for setting values at boundaries
    bool condX[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                     grid_x + 2 < x_lim};
    bool condY[4] = {grid_y - 1 >= 0, true, grid_y + 1 < y_lim,
                     grid_y + 2 < y_lim};
    int offX[4]   = {condX[0] ? -1 : 0, 0, condX[2] ? 1 : 0,
                   condX[3] ? 2 : (condX[2] ? 1 : 0)};
    int offY[4]   = {condY[0] ? -1 : 0, 0, condY[2] ? 1 : 0,
                   condY[3] ? 2 : (condY[2] ? 1 : 0)};

    InterpValTy zero = ZERO;
    for (int n = 0; n < batch; n++) {
        int idx_n = idx + n * in.strides[batch_dim];
        // for bicubic interpolation, work with 4x4 val at a time
        InterpValTy val[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int ioff_j = idx_n + offY[j] * y_stride;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                bool cond = (doclamp || (condX[i] && condY[j]));
                val[j][i] = cond ? d_in[ioff_j + offX[i] * x_stride] : zero;
            }
        }
        bool spline = method == AF_INTERP_CUBIC_SPLINE ||
                      method == AF_INTERP_BICUBIC_SPLINE;
        d_out[ooff + n * out.strides[batch_dim]] =
            bicubicInterpFunc(val, off_x, off_y, spline);
    }
}
#endif
#endif
