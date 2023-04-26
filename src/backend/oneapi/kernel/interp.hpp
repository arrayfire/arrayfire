/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <kernel/accessors.hpp>
#include <math.hpp>
#include <types.hpp>
#include <af/constants.h>

#include <sycl/sycl.hpp>

#include <algorithm>

namespace arrayfire {
namespace oneapi {

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
Ty linearInterpFunc(Ty val[2], Tp ratio) {
    return (1 - ratio) * val[0] + ratio * val[1];
}

template<typename Ty, typename Tp>
Ty bilinearInterpFunc(Ty val[2][2], Tp xratio, Tp yratio) {
    Ty res[2];
    res[0] = linearInterpFunc(val[0], xratio);
    res[1] = linearInterpFunc(val[1], xratio);
    return linearInterpFunc(res, yratio);
}

template<typename Ty, typename Tp>
inline static Ty cubicInterpFunc(Ty val[4], Tp xratio, bool spline) {
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
inline static Ty bicubicInterpFunc(Ty val[4][4], Tp xratio, Tp yratio,
                                   bool spline) {
    Ty res[4];
    res[0] = cubicInterpFunc(val[0], xratio, spline);
    res[1] = cubicInterpFunc(val[1], xratio, spline);
    res[2] = cubicInterpFunc(val[2], xratio, spline);
    res[3] = cubicInterpFunc(val[3], xratio, spline);
    return cubicInterpFunc(res, yratio, spline);
}

template<typename Ty, typename Tp, int order>
struct Interp1 {};

template<typename Ty, typename Tp>
struct Interp1<Ty, Tp, 1> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x,
                    int xdim, af::interpType method, int batch, bool clamp,
                    int batch_dim = 1) {
        Ty zero = scalar<Ty>(0);

        const int x_lim    = iInfo.dims[xdim];
        const int x_stride = iInfo.strides[xdim];

        int xid = (method == AF_INTERP_LOWER ? sycl::floor(x) : sycl::round(x));
        bool cond = xid >= 0 && xid < x_lim;
        if (clamp) xid = sycl::max((int)0, sycl::min(xid, x_lim));

        const int idx = ioff + xid * x_stride;

        for (int n = 0; n < batch; n++) {
            Ty outval =
                (cond || clamp) ? in[idx + n * iInfo.strides[batch_dim]] : zero;
            out[ooff + n * oInfo.strides[batch_dim]] = outval;
        }
    }
};

template<typename Ty, typename Tp>
struct Interp1<Ty, Tp, 2> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x,
                    int xdim, af::interpType method, int batch, bool clamp,
                    int batch_dim = 1) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = sycl::floor(x);  // nearest grid
        const WT off_x   = x - grid_x;      // fractional offset

        const int x_lim    = iInfo.dims[xdim];
        const int x_stride = iInfo.strides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[2] = {true, grid_x + 1 < x_lim};
        int offx[2]  = {0, cond[1] ? 1 : 0};
        WT ratio     = off_x;
        if (method == AF_INTERP_LINEAR_COSINE) {
            // Smooth the factional part with cosine
            ratio = (1 - sycl::cospi(ratio)) / 2;
        }

        Ty zero = scalar<Ty>(0);

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * iInfo.strides[batch_dim];
            VT val[2] = {
                (clamp || cond[0]) ? in[idx_n + offx[0] * x_stride] : zero,
                (clamp || cond[1]) ? in[idx_n + offx[1] * x_stride] : zero};
            out[ooff + n * oInfo.strides[batch_dim]] =
                linearInterpFunc(val, ratio);
        }
    }
};

template<typename Ty, typename Tp>
struct Interp1<Ty, Tp, 3> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x,
                    int xdim, af::interpType method, int batch, bool clamp,
                    int batch_dim = 1) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = sycl::floor(x);  // nearest grid
        const WT off_x   = x - grid_x;      // fractional offset

        const int x_lim    = iInfo.dims[xdim];
        const int x_stride = iInfo.strides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                        grid_x + 2 < x_lim};
        int offx[4]  = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                       cond[3] ? 2 : (cond[2] ? 1 : 0)};

        bool spline = method == AF_INTERP_CUBIC_SPLINE;
        Ty zero     = scalar<Ty>(0);
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * iInfo.strides[batch_dim];
            VT val[4];
            for (int i = 0; i < 4; i++) {
                val[i] =
                    (clamp || cond[i]) ? in[idx_n + offx[i] * x_stride] : zero;
            }
            out[ooff + n * oInfo.strides[batch_dim]] =
                cubicInterpFunc(val, off_x, spline);
        }
    }
};

template<typename Ty, typename Tp, int order>
struct Interp2 {};

template<typename Ty, typename Tp>
struct Interp2<Ty, Tp, 1> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x, Tp y,
                    int xdim, int ydim, af::interpType method, int batch,
                    bool clamp, int batch_dim = 2) {
        int xid = (method == AF_INTERP_LOWER ? sycl::floor(x) : sycl::round(x));
        int yid = (method == AF_INTERP_LOWER ? sycl::floor(y) : sycl::round(y));

        const int x_lim    = iInfo.dims[xdim];
        const int y_lim    = iInfo.dims[ydim];
        const int x_stride = iInfo.strides[xdim];
        const int y_stride = iInfo.strides[ydim];

        if (clamp) {
            xid = sycl::max(0, sycl::min(xid, (int)iInfo.dims[xdim]));
            yid = sycl::max(0, sycl::min(yid, (int)iInfo.dims[ydim]));
        }

        const int idx = ioff + yid * y_stride + xid * x_stride;

        bool condX = xid >= 0 && xid < x_lim;
        bool condY = yid >= 0 && yid < y_lim;

        Ty zero   = scalar<Ty>(0);
        bool cond = condX && condY;

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * iInfo.strides[batch_dim];
            Ty val    = (clamp || cond) ? in[idx_n] : zero;
            out[ooff + n * oInfo.strides[batch_dim]] = val;
        }
    }
};

template<typename Ty, typename Tp>
struct Interp2<Ty, Tp, 2> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x, Tp y,
                    int xdim, int ydim, af::interpType method, int batch,
                    bool clamp, int batch_dim = 2) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = sycl::floor(x);
        const WT off_x   = x - grid_x;

        const int grid_y = sycl::floor(y);
        const WT off_y   = y - grid_y;

        const int x_lim    = iInfo.dims[xdim];
        const int y_lim    = iInfo.dims[ydim];
        const int x_stride = iInfo.strides[xdim];
        const int y_stride = iInfo.strides[ydim];
        const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

        bool condX[2] = {true, x + 1 < x_lim};
        bool condY[2] = {true, y + 1 < y_lim};
        int offx[2]   = {0, condX[1] ? 1 : 0};
        int offy[2]   = {0, condY[1] ? 1 : 0};

        WT xratio = off_x, yratio = off_y;
        if (method == AF_INTERP_LINEAR_COSINE ||
            method == AF_INTERP_BILINEAR_COSINE) {
            // Smooth the factional part with cosine
            xratio = (1 - sycl::cospi(xratio)) / 2;
            yratio = (1 - sycl::cospi(yratio)) / 2;
        }

        Ty zero = scalar<Ty>(0);

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * iInfo.strides[batch_dim];
            VT val[2][2];
            for (int j = 0; j < 2; j++) {
                int ioff_j = idx_n + offy[j] * y_stride;
                for (int i = 0; i < 2; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] = (cond) ? in[ioff_j + offx[i] * x_stride] : zero;
                }
            }
            out[ooff + n * oInfo.strides[batch_dim]] =
                bilinearInterpFunc(val, xratio, yratio);
        }
    }
};

template<typename Ty, typename Tp>
struct Interp2<Ty, Tp, 3> {
    void operator()(write_accessor<Ty> out, KParam oInfo, int ooff,
                    read_accessor<Ty> in, KParam iInfo, int ioff, Tp x, Tp y,
                    int xdim, int ydim, af::interpType method, int batch,
                    bool clamp, int batch_dim = 2) {
        typedef typename itype_t<Tp>::wtype WT;
        typedef typename itype_t<Ty>::vtype VT;

        const int grid_x = sycl::floor(x);
        const WT off_x   = x - grid_x;

        const int grid_y = sycl::floor(y);
        const WT off_y   = y - grid_y;

        const int x_lim    = iInfo.dims[xdim];
        const int y_lim    = iInfo.dims[ydim];
        const int x_stride = iInfo.strides[xdim];
        const int y_stride = iInfo.strides[ydim];
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
            int idx_n = idx + n * iInfo.strides[batch_dim];
            VT val[4][4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int ioff_j = idx_n + offY[j] * y_stride;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] = (cond) ? in[ioff_j + offX[i] * x_stride] : zero;
                }
            }

            out[ooff + n * oInfo.strides[batch_dim]] =
                bicubicInterpFunc(val, off_x, off_y, spline);
        }
    }
};

}  // namespace oneapi
}  // namespace arrayfire
