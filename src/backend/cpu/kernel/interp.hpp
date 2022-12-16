/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <Param.hpp>
#include <common/complex.hpp>
#include <math.hpp>
#include <af/constants.h>
#include <type_traits>

namespace arrayfire {
namespace cpu {
namespace kernel {

using std::conditional;
using std::is_same;

template<typename T>
using wtype_t =
    typename conditional<is_same<T, double>::value, double, float>::type;

template<typename T>
using vtype_t =
    typename conditional<common::is_complex<T>::value, T, wtype_t<T>>::type;

template<typename InT, typename LocT>
InT linearInterpFunc(InT val[2], LocT ratio) {
    return (1 - ratio) * val[0] + ratio * val[1];
}

template<typename InT, typename LocT>
InT bilinearInterpFunc(InT val[2][2], LocT xratio, LocT yratio) {
    InT res[2];
    res[0] = linearInterpFunc(val[0], xratio);
    res[1] = linearInterpFunc(val[1], xratio);
    return linearInterpFunc(res, yratio);
}

template<typename InT, typename LocT>
InT cubicInterpFunc(InT val[4], LocT xratio, bool spline) {
    InT a0, a1, a2, a3;
    if (spline) {
        a0 = scalar<InT>(-0.5) * val[0] + scalar<InT>(1.5) * val[1] +
             scalar<InT>(-1.5) * val[2] + scalar<InT>(0.5) * val[3];

        a1 = scalar<InT>(1.0) * val[0] + scalar<InT>(-2.5) * val[1] +
             scalar<InT>(2.0) * val[2] + scalar<InT>(-0.5) * val[3];

        a2 = scalar<InT>(-0.5) * val[0] + scalar<InT>(0.5) * val[2];

        a3 = val[1];
    } else {
        a0 = val[3] - val[2] - val[0] + val[1];
        a1 = val[0] - val[1] - a0;
        a2 = val[2] - val[0];
        a3 = val[1];
    }

    LocT xratio2 = xratio * xratio;
    LocT xratio3 = xratio2 * xratio;

    return a0 * xratio3 + a1 * xratio2 + a2 * xratio + a3;
}

template<typename InT, typename LocT>
InT bicubicInterpFunc(InT val[4][4], LocT xratio, LocT yratio, bool spline) {
    InT res[4];
    res[0] = cubicInterpFunc(val[0], xratio, spline);
    res[1] = cubicInterpFunc(val[1], xratio, spline);
    res[2] = cubicInterpFunc(val[2], xratio, spline);
    res[3] = cubicInterpFunc(val[3], xratio, spline);
    return cubicInterpFunc(res, yratio, spline);
}

template<typename InT, typename LocT, int order>
struct Interp1 {};

template<typename InT, typename LocT>
struct Interp1<InT, LocT, 1> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, af_interp_type method, int batch, bool clamp,
                    int xdim = 0, int batch_dim = 1) {
        const InT *inptr    = in.get();
        const dim4 idims    = in.dims();
        const dim4 istrides = in.strides();

        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        const int x_lim    = idims[xdim];
        const int x_stride = istrides[xdim];

        int xid   = (method == AF_INTERP_LOWER ? std::floor(x) : std::round(x));
        bool cond = xid >= 0 && xid < x_lim;
        if (clamp) xid = std::max(0, std::min(xid, x_lim));

        const int idx = ioff + xid * x_stride;

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * istrides[batch_dim];
            outptr[ooff + n * ostrides[batch_dim]] =
                (cond || clamp) ? inptr[idx_n] : scalar<InT>(0);
        }
    }
};

template<typename InT, typename LocT>
struct Interp1<InT, LocT, 2> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, af_interp_type method, int batch, bool clamp,
                    int xdim = 0, int batch_dim = 1) {
        typedef vtype_t<InT> VT;

        const int grid_x = floor(x);    // nearest grid
        const LocT off_x = x - grid_x;  // fractional offset

        const InT *inptr    = in.get();
        const dim4 idims    = in.dims();
        const dim4 istrides = in.strides();
        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        const int x_lim    = idims[xdim];
        const int x_stride = istrides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[2] = {true, grid_x + 1 < x_lim};
        int offx[2]  = {0, cond[1] ? 1 : 0};

        LocT ratio = off_x;
        if (method == AF_INTERP_LINEAR_COSINE) {
            // Smooth the factional part with cosine
            ratio = (1 - std::cos(ratio * af::Pi)) / 2;
        }

        const VT zero = scalar<VT>(0);
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * istrides[batch_dim];
            VT val[2] = {zero, zero};
            for (int i = 0; i < 2; i++) {
                if (clamp || cond[i])
                    val[i] = inptr[idx_n + offx[i] * x_stride];
            }
            outptr[ooff + n * ostrides[batch_dim]] =
                linearInterpFunc(val, ratio);
        }
    }
};

template<typename InT, typename LocT>
struct Interp1<InT, LocT, 3> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, af_interp_type method, int batch, bool clamp,
                    int xdim = 0, int batch_dim = 1) {
        typedef vtype_t<InT> VT;

        const int grid_x = floor(x);    // nearest grid
        const LocT off_x = x - grid_x;  // fractional offset

        const InT *inptr    = in.get();
        const dim4 idims    = in.dims();
        const dim4 istrides = in.strides();
        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        const int x_lim    = idims[xdim];
        const int x_stride = istrides[xdim];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                        grid_x + 2 < x_lim};
        int off[4]   = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                      cond[3] ? 2 : (cond[2] ? 1 : 0)};

        const VT zero = scalar<VT>(0);
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * istrides[batch_dim];
            VT val[4] = {zero, zero, zero, zero};
            for (int i = 0; i < 4; i++) {
                if (clamp || cond[i]) val[i] = inptr[idx_n + off[i] * x_stride];
            }
            bool spline = method == AF_INTERP_CUBIC_SPLINE;
            outptr[ooff + n * ostrides[batch_dim]] =
                cubicInterpFunc(val, off_x, spline);
        }
    }
};

template<typename InT, typename LocT, int order>
struct Interp2 {};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, 1> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, LocT y, af_interp_type method, int nimages,
                    bool clamp, int xdim = 0, int ydim = 1, int batch_dim = 2) {
        const InT *inptr    = in.get();
        const dim4 istrides = in.strides();
        const dim4 idims    = in.dims();

        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        int xid = (method == AF_INTERP_LOWER ? std::floor(x) : std::round(x));
        int yid = (method == AF_INTERP_LOWER ? std::floor(y) : std::round(y));

        const int x_lim    = idims[xdim];
        const int y_lim    = idims[ydim];
        const int x_stride = istrides[xdim];
        const int y_stride = istrides[ydim];
        const int idx      = ioff + yid * y_stride + xid * x_stride;

        bool condX = xid >= 0 && xid < x_lim;
        bool condY = yid >= 0 && yid < y_lim;

        if (clamp) {
            xid = std::max(0, std::min(xid, x_lim));
            yid = std::max(0, std::min(yid, y_lim));
        }

        bool cond = condX && condY;
        for (int n = 0; n < nimages; n++) {
            int idx_n = idx + n * istrides[batch_dim];
            outptr[ooff + n * ostrides[batch_dim]] =
                (clamp || cond) ? inptr[idx_n] : scalar<InT>(0);
        }
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, 2> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, LocT y, af_interp_type method, int nimages,
                    bool clamp, int xdim = 0, int ydim = 1, int batch_dim = 2) {
        typedef vtype_t<InT> VT;

        const InT *inptr    = in.get();
        const dim4 idims    = in.dims();
        const dim4 istrides = in.strides();

        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        const int grid_x = floor(x);
        const LocT off_x = x - grid_x;

        const int grid_y = floor(y);
        const LocT off_y = y - grid_y;

        const int x_lim    = idims[xdim];
        const int y_lim    = idims[ydim];
        const int x_stride = istrides[xdim];
        const int y_stride = istrides[ydim];
        const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

        bool condX[2] = {true, x + 1 < x_lim};
        bool condY[2] = {true, y + 1 < y_lim};

        int offX[2] = {0, condX[1] ? 1 : 0};
        int offY[2] = {0, condY[1] ? 1 : 0};

        VT zero = scalar<VT>(0);

        LocT xratio = off_x, yratio = off_y;
        if (method == AF_INTERP_LINEAR_COSINE ||
            method == AF_INTERP_BILINEAR_COSINE) {
            // Smooth the factional part with cosine
            xratio = (1 - std::cos(xratio * af::Pi)) / 2;
            yratio = (1 - std::cos(yratio * af::Pi)) / 2;
        }

        for (int n = 0; n < nimages; n++) {
            int idx_n = idx + n * istrides[batch_dim];
            VT val[2][2];
            for (int j = 0; j < 2; j++) {
                int off_y = idx_n + offY[j] * y_stride;
                for (int i = 0; i < 2; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] = cond ? inptr[off_y + offX[i] * x_stride] : zero;
                }
            }
            outptr[ooff + n * ostrides[batch_dim]] =
                bilinearInterpFunc(val, off_x, off_y);
        }
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, 3> {
    void operator()(Param<InT> &out, int ooff, CParam<InT> &in, int ioff,
                    LocT x, LocT y, af_interp_type method, int nimages,
                    bool clamp, int xdim = 0, int ydim = 1, int batch_dim = 2) {
        typedef vtype_t<InT> VT;

        const InT *inptr    = in.get();
        const dim4 idims    = in.dims();
        const dim4 istrides = in.strides();

        InT *outptr         = out.get();
        const dim4 ostrides = out.strides();

        const int grid_x = floor(x);
        const LocT off_x = x - grid_x;

        const int grid_y = floor(y);
        const LocT off_y = y - grid_y;

        const int x_lim    = idims[xdim];
        const int y_lim    = idims[ydim];
        const int x_stride = istrides[xdim];
        const int y_stride = istrides[ydim];
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

        bool spline = (method == AF_INTERP_CUBIC_SPLINE ||
                       method == AF_INTERP_BICUBIC_SPLINE);
        VT zero     = scalar<VT>(0);
        for (int n = 0; n < nimages; n++) {
            int idx_n = idx + n * istrides[batch_dim];

            // for bicubic interpolation, work with 4x4 val at a time
            VT val[4][4];
            for (int j = 0; j < 4; j++) {
                int ioff_j = idx_n + offY[j] * y_stride;
                for (int i = 0; i < 4; i++) {
                    bool cond = clamp || (condX[i] && condY[j]);
                    val[j][i] =
                        cond ? inptr[ioff_j + offX[i] * x_stride] : zero;
                }
            }
            outptr[ooff + n * ostrides[batch_dim]] =
                bicubicInterpFunc(val, off_x, off_y, spline);
        }
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
