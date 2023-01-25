/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
struct itype_t {
    typedef float wtype;
    typedef float vtype;
};

template<>
struct itype_t<double> {
    typedef double wtype;
    typedef double vtype;
};

template<>
struct itype_t<cfloat> {
    typedef float wtype;
    typedef cfloat vtype;
};

template<>
struct itype_t<cdouble> {
    typedef double wtype;
    typedef cdouble vtype;
};

template<typename Ty, typename Tp>
__device__ Ty linearInterpFunc(Ty val[2], Tp ratio) {
    return (1 - ratio) * val[0] + ratio * val[1];
}

template<typename Ty, typename Tp>
__device__ Ty bilinearInterpFunc(Ty val[2][2], Tp xratio, Tp yratio) {
    Ty res[2];
    res[0] = linearInterpFunc(val[0], xratio);
    res[1] = linearInterpFunc(val[1], xratio);
    return linearInterpFunc(res, yratio);
}

template<typename Ty, typename Tp>
__device__ inline static Ty cubicInterpFunc(Ty val[4], Tp xratio, bool spline) {
    Ty a0, a1, a2, a3;
    if (spline) {
        a0 = scalar<Ty>(-0.5) * val[0] + scalar<Ty>(1.5) * val[1] +
             scalar<Ty>(-1.5) * val[2] + scalar<Ty>(0.5) * val[3];

        a1 = scalar<Ty>(1.0) * val[0] + scalar<Ty>(-2.5) * val[1] +
             scalar<Ty>(2.0) * val[2] + scalar<Ty>(-0.5) * val[3];

        a2 = scalar<Ty>(-0.5) * val[0] + scalar<Ty>(0.5) * val[2];

        a3 = val[1];
    } else {
        a0 = val[3] - val[2] - val[0] + val[1];
        a1 = val[0] - val[1] - a0;
        a2 = val[2] - val[0];
        a3 = val[1];
    }

    Tp xratio2 = xratio * xratio;
    Tp xratio3 = xratio2 * xratio;

    return a0 * xratio3 + a1 * xratio2 + a2 * xratio + a3;
}

template<typename Ty, typename Tp>
__device__ inline static Ty bicubicInterpFunc(Ty val[4][4], Tp xratio,
                                              Tp yratio, bool spline) {
    Ty res[4];
    res[0] = cubicInterpFunc(val[0], xratio, spline);
    res[1] = cubicInterpFunc(val[1], xratio, spline);
    res[2] = cubicInterpFunc(val[2], xratio, spline);
    res[3] = cubicInterpFunc(val[3], xratio, spline);
    return cubicInterpFunc(res, yratio, spline);
}

template<typename Ty, typename Tp, int xdim, int order>
struct Interp1 {};

template<typename Ty, typename Tp, int xdim>
struct Interp1<Ty, Tp, xdim, 1> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, af::interpType method, int batch,
                               bool clamp, int batch_dim = 1) {
        Ty zero = scalar<Ty>(0);

        const int x_lim    = in.dims[xdim];
        const int x_stride = in.strides[xdim];

        int xid   = (method == AF_INTERP_LOWER ? floor(x) : round(x));
        bool cond = xid >= 0 && xid < x_lim;
        if (clamp) xid = max(0, min(xid, x_lim));

        const int idx = ioff + xid * x_stride;

        for (int n = 0; n < batch; n++) {
            Ty outval                                  = (cond || clamp)
                                                             ? in.ptr[idx + n * in.strides[batch_dim]]
                                                             : zero;
            out.ptr[ooff + n * out.strides[batch_dim]] = outval;
        }
    }
};

template<typename Ty, typename Tp, int xdim>
struct Interp1<Ty, Tp, xdim, 2> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, af::interpType method, int batch,
                               bool clamp, int batch_dim = 1) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = floor(x);    // nearest grid
        const WT off_x   = x - grid_x;  // fractional offset

        const int x_lim    = in.dims[xdim];
        const int x_stride = in.strides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[2] = {true, grid_x + 1 < x_lim};
        int offx[2]  = {0, cond[1] ? 1 : 0};
        WT ratio     = off_x;
        if (method == AF_INTERP_LINEAR_COSINE) {
            // Smooth the factional part with cosine
            ratio = (1 - cos(ratio * CUDART_PI)) / 2;
        }

        Ty zero = scalar<Ty>(0);

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            VT val[2] = {
                (clamp || cond[0]) ? in.ptr[idx_n + offx[0] * x_stride] : zero,
                (clamp || cond[1]) ? in.ptr[idx_n + offx[1] * x_stride] : zero};
            out.ptr[ooff + n * out.strides[batch_dim]] =
                linearInterpFunc(val, ratio);
        }
    }
};

template<typename Ty, typename Tp, int xdim>
struct Interp1<Ty, Tp, xdim, 3> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, af::interpType method, int batch,
                               bool clamp, int batch_dim = 1) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = floor(x);    // nearest grid
        const WT off_x   = x - grid_x;  // fractional offset

        const int x_lim    = in.dims[xdim];
        const int x_stride = in.strides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                        grid_x + 2 < x_lim};
        int offx[4]  = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                       cond[3] ? 2 : (cond[2] ? 1 : 0)};

        bool spline = method == AF_INTERP_CUBIC_SPLINE;
        Ty zero     = scalar<Ty>(0);
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            VT val[4];
            for (int i = 0; i < 4; i++) {
                val[i] = (clamp || cond[i]) ? in.ptr[idx_n + offx[i] * x_stride]
                                            : zero;
            }
            out.ptr[ooff + n * out.strides[batch_dim]] =
                cubicInterpFunc(val, off_x, spline);
        }
    }
};

template<typename Ty, typename Tp, int xdim, int ydim, int order>
struct Interp2 {};

template<typename Ty, typename Tp, int xdim, int ydim>
struct Interp2<Ty, Tp, xdim, ydim, 1> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, Tp y, af::interpType method, int batch,
                               bool clamp, int batch_dim = 2) {
        int xid = (method == AF_INTERP_LOWER ? floor(x) : round(x));
        int yid = (method == AF_INTERP_LOWER ? floor(y) : round(y));

        const int x_lim    = in.dims[xdim];
        const int y_lim    = in.dims[ydim];
        const int x_stride = in.strides[xdim];
        const int y_stride = in.strides[ydim];

        if (clamp) {
            xid = max(0, min(xid, in.dims[xdim]));
            yid = max(0, min(yid, in.dims[ydim]));
        }

        const int idx = ioff + yid * y_stride + xid * x_stride;

        bool condX = xid >= 0 && xid < x_lim;
        bool condY = yid >= 0 && yid < y_lim;

        Ty zero   = scalar<Ty>(0);
        bool cond = condX && condY;

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            Ty val    = (clamp || cond) ? in.ptr[idx_n] : zero;
            out.ptr[ooff + n * out.strides[batch_dim]] = val;
        }
    }
};

template<typename Ty, typename Tp, int xdim, int ydim>
struct Interp2<Ty, Tp, xdim, ydim, 2> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, Tp y, af::interpType method, int batch,
                               bool clamp, int batch_dim = 2) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = floor(x);
        const WT off_x   = x - grid_x;

        const int grid_y = floor(y);
        const WT off_y   = y - grid_y;

        const int x_lim    = in.dims[xdim];
        const int y_lim    = in.dims[ydim];
        const int x_stride = in.strides[xdim];
        const int y_stride = in.strides[ydim];
        const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

        bool condX[2] = {true, x + 1 < x_lim};
        bool condY[2] = {true, y + 1 < y_lim};
        int offx[2]   = {0, condX[1] ? 1 : 0};
        int offy[2]   = {0, condY[1] ? 1 : 0};

        WT xratio = off_x, yratio = off_y;
        if (method == AF_INTERP_LINEAR_COSINE ||
            method == AF_INTERP_BILINEAR_COSINE) {
            // Smooth the factional part with cosine
            xratio = (1 - cos(xratio * CUDART_PI)) / 2;
            yratio = (1 - cos(yratio * CUDART_PI)) / 2;
        }

        Ty zero = scalar<Ty>(0);

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            VT val[2][2];
            for (int j = 0; j < 2; j++) {
                int ioff_j = idx_n + offy[j] * y_stride;
                for (int i = 0; i < 2; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] =
                        (cond) ? in.ptr[ioff_j + offx[i] * x_stride] : zero;
                }
            }
            out.ptr[ooff + n * out.strides[batch_dim]] =
                bilinearInterpFunc(val, xratio, yratio);
        }
    }
};

template<typename Ty, typename Tp, int xdim, int ydim>
struct Interp2<Ty, Tp, xdim, ydim, 3> {
    __device__ void operator()(Param<Ty> out, int ooff, CParam<Ty> in, int ioff,
                               Tp x, Tp y, af::interpType method, int batch,
                               bool clamp, int batch_dim = 2) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = floor(x);
        const WT off_x   = x - grid_x;

        const int grid_y = floor(y);
        const WT off_y   = y - grid_y;

        const int x_lim    = in.dims[xdim];
        const int y_lim    = in.dims[ydim];
        const int x_stride = in.strides[xdim];
        const int y_stride = in.strides[ydim];
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

        // for bicubic interpolation, work with 4x4 val at a time
        Ty zero     = scalar<Ty>(0);
        bool spline = (method == AF_INTERP_CUBIC_SPLINE ||
                       method == AF_INTERP_BICUBIC_SPLINE);
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            VT val[4][4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int ioff_j = idx_n + offY[j] * y_stride;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] =
                        (cond) ? in.ptr[ioff_j + offX[i] * x_stride] : zero;
                }
            }

            out.ptr[ooff + n * out.strides[batch_dim]] =
                bicubicInterpFunc(val, off_x, off_y, spline);
        }
    }
};

}  // namespace cuda
}  // namespace arrayfire
