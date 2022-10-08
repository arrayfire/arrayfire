/*******************************************************
 * Copyright (c) 2022 ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = 256 / TILE_DIM;

template<typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write,
                   sycl::access::target::local>;

template<typename Ty, typename Tp>
class approx1Kernel {
   public:
    approx1Kernel(sycl::accessor<Ty> d_yo, const KParam yo,
                  sycl::accessor<Ty> d_yi, const KParam yi,
                  sycl::accessor<Tp> d_xo, const KParam xo, const Tp xi_beg,
                  const Tp xi_step_reproc, const Ty offGrid,
                  const int blocksMatX, const int batch, const int method,
                  const int XDIM, const int INTERP_ORDER)
        : d_yo_(d_yo)
        , yo_(yo)
        , d_yi_(d_yi)
        , yi_(yi)
        , d_xo_(d_xo)
        , xo_(xo)
        , xi_beg_(xi_beg)
        , xi_step_reproc_(xi_step_reproc)
        , offGrid_(offGrid)
        , blocksMatX_(blocksMatX)
        , batch_(batch)
        , method_(method)
        , XDIM_(XDIM)
        , INTERP_ORDER_(INTERP_ORDER) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int idw = g.get_group_id(1) / yo_.dims[2];
        const int idz = g.get_group_id(1) - idw * yo_.dims[2];

        const int idy        = g.get_group_id(0) / blocksMatX_;
        const int blockIdx_x = g.get_group_id(0) - idy * blocksMatX_;
        const int idx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);

        if (idx >= yo_.dims[0] || idy >= yo_.dims[1] || idz >= yo_.dims[2] ||
            idw >= yo_.dims[3])
            return;

        // FIXME: Only cubic interpolation is doing clamping
        // We need to make it consistent across all methods
        // Not changing the behavior because tests will fail
        const bool doclamp = INTERP_ORDER_ == 3;

        bool is_off[] = {xo_.dims[0] > 1, xo_.dims[1] > 1, xo_.dims[2] > 1,
                         xo_.dims[3] > 1};

        const int yo_idx = idw * yo_.strides[3] + idz * yo_.strides[2] +
                           idy * yo_.strides[1] + idx + yo_.offset;

        int xo_idx = idx * is_off[0] + xo_.offset;
        if (batch_) {
            xo_idx += idw * xo_.strides[3] * is_off[3];
            xo_idx += idz * xo_.strides[2] * is_off[2];
            xo_idx += idy * xo_.strides[1] * is_off[1];
        }

        const Tp x = (d_xo_[xo_idx] - xi_beg_) * xi_step_reproc_;

#pragma unroll
        for (int flagIdx = 0; flagIdx < 4; ++flagIdx) {
            is_off[flagIdx] = true;
        }
        is_off[XDIM_] = false;

        if (x < 0 || yi_.dims[XDIM_] < x + 1) {
            d_yo_[yo_idx] = offGrid_;
            return;
        }

        int yi_idx = idx * is_off[0] + yi_.offset;
        yi_idx += idw * yi_.strides[3] * is_off[3];
        yi_idx += idz * yi_.strides[2] * is_off[2];
        yi_idx += idy * yi_.strides[1] * is_off[1];

        if (INTERP_ORDER_ == 1)
            interp1o1(d_yo_, yo_, yo_idx, d_yi_, yi_, yi_idx, x, method_, 1,
                      doclamp, 1);
        if (INTERP_ORDER_ == 2)
            interp1o2(d_yo_, yo_, yo_idx, d_yi_, yi_, yi_idx, x, method_, 1,
                      doclamp, 1);
        if (INTERP_ORDER_ == 3)
            interp1o3(d_yo_, yo_, yo_idx, d_yi_, yi_, yi_idx, x, method_, 1,
                      doclamp, 1);
    }

    void interp1o1(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x,
                   int method, int batch, bool doclamp, int batch_dim) const {
        Ty zero = (Ty)0;

        const int x_lim    = in.dims[XDIM_];
        const int x_stride = in.strides[XDIM_];

        int xid   = (method == AF_INTERP_LOWER ? floor(x) : round(x));
        bool cond = xid >= 0 && xid < x_lim;
        if (doclamp) xid = fmax(0, fmin(xid, x_lim));

        const int idx = ioff + xid * x_stride;

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            d_out[ooff + n * out.strides[batch_dim]] =
                (doclamp || cond) ? d_in[idx_n] : zero;
        }
    }

#if IS_CPLX
#if USE_DOUBLE
    typedef double ScalarTy;
#else
    typedef float ScalarTy;
#endif
    Ty __mulrc(ScalarTy s, Ty v) {
        InterpInTy out = {s * v.x, s * v.y};
        return out;
    }
#define MULRC(a, b) __mulrc(a, b)
#define MULCR(a, b) __mulrc(b, a)
#else
#define MULRC(a, b) (a) * (b)
#define MULCR(a, b) (a) * (b)
#endif

    Ty linearInterpFunc(Ty val[2], Tp ratio) const {
        return MULRC((1 - ratio), val[0]) + MULRC(ratio, val[1]);
    }

    Ty bilinearInterpFunc(Ty val[2][2], Tp xratio, Tp yratio) const {
        Ty res[2];
        res[0] = linearInterpFunc(val[0], xratio);
        res[1] = linearInterpFunc(val[1], xratio);
        return linearInterpFunc(res, yratio);
    }

    Ty cubicInterpFunc(Ty val[4], Tp xratio, bool spline) const {
        Ty a0, a1, a2, a3;
        if (spline) {
            a0 = MULRC((Tp)-0.5, val[0]) + MULRC((Tp)1.5, val[1]) +
                 MULRC((Tp)-1.5, val[2]) + MULRC((Tp)0.5, val[3]);

            a1 = MULRC((Tp)1.0, val[0]) + MULRC((Tp)-2.5, val[1]) +
                 MULRC((Tp)2.0, val[2]) + MULRC((Tp)-0.5, val[3]);

            a2 = MULRC((Tp)-0.5, val[0]) + MULRC((Tp)0.5, val[2]);

            a3 = val[1];
        } else {
            a0 = val[3] - val[2] - val[0] + val[1];
            a1 = val[0] - val[1] - a0;
            a2 = val[2] - val[0];
            a3 = val[1];
        }

        Tp xratio2 = xratio * xratio;
        Tp xratio3 = xratio2 * xratio;

        return MULCR(a0, xratio3) + MULCR(a1, xratio2) + MULCR(a2, xratio) + a3;
    }

    Ty bicubicInterpFunc(Ty val[4][4], Tp xratio, Tp yratio,
                         bool spline) const {
        Ty res[4];
        res[0] = cubicInterpFunc(val[0], xratio, spline);
        res[1] = cubicInterpFunc(val[1], xratio, spline);
        res[2] = cubicInterpFunc(val[2], xratio, spline);
        res[3] = cubicInterpFunc(val[3], xratio, spline);
        return cubicInterpFunc(res, yratio, spline);
    }

    void interp1o2(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x,
                   int method, int batch, bool doclamp, int batch_dim) const {
        const int grid_x = floor(x);    // nearest grid
        const Tp off_x   = x - grid_x;  // fractional offset

        const int x_lim    = in.dims[XDIM_];
        const int x_stride = in.strides[XDIM_];
        const int idx      = ioff + grid_x * x_stride;

        Ty zero      = (Ty)0;
        bool cond[2] = {true, grid_x + 1 < x_lim};
        int offx[2]  = {0, cond[1] ? 1 : 0};
        Tp ratio     = off_x;
        if (method == AF_INTERP_LINEAR_COSINE) {
            ratio = (1 - cos(ratio * (Tp)M_PI)) / 2;
        }

        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            Ty val[2] = {
                (doclamp || cond[0]) ? d_in[idx_n + offx[0] * x_stride] : zero,
                (doclamp || cond[1]) ? d_in[idx_n + offx[1] * x_stride] : zero};

            d_out[ooff + n * out.strides[batch_dim]] =
                linearInterpFunc(val, ratio);
        }
    }

    void interp1o3(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x,
                   int method, int batch, bool doclamp, int batch_dim) const {
        const int grid_x = floor(x);    // nearest grid
        const Tp off_x   = x - grid_x;  // fractional offset

        const int x_lim    = in.dims[XDIM_];
        const int x_stride = in.strides[XDIM_];
        const int idx      = ioff + grid_x * x_stride;

        bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < x_lim,
                        grid_x + 2 < x_lim};
        int off[4]   = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0,
                      cond[3] ? 2 : (cond[2] ? 1 : 0)};

        Ty zero = (Ty)0;

        for (int n = 0; n < batch; n++) {
            Ty val[4];
            int idx_n = idx + n * in.strides[batch_dim];
            for (int i = 0; i < 4; i++) {
                val[i] = (doclamp || cond[i]) ? d_in[idx_n + off[i] * x_stride]
                                              : zero;
            }
            bool spline = method == AF_INTERP_CUBIC_SPLINE;
            d_out[ooff + n * out.strides[batch_dim]] =
                cubicInterpFunc(val, off_x, spline);
        }
    }

   private:
    sycl::accessor<Ty> d_yo_;
    const KParam yo_;
    sycl::accessor<Ty> d_yi_;
    const KParam yi_;
    sycl::accessor<Tp> d_xo_;
    const KParam xo_;
    const Tp xi_beg_;
    const Tp xi_step_reproc_;
    const Ty offGrid_;
    const int blocksMatX_;
    const int batch_;
    const int method_;
    const int XDIM_;
    const int INTERP_ORDER_;
};

template<typename Ty, typename Tp>
void approx1(Param<Ty> yo, const Param<Ty> yi, const Param<Tp> xo,
             const int xdim, const Tp xi_beg, const Tp xi_step,
             const float offGrid, const af_interp_type method,
             const int order) {
    constexpr int THREADS = 256;

    auto local         = sycl::range{THREADS, 1};
    dim_t blocksPerMat = divup(yo.info.dims[0], local[0]);
    auto global        = sycl::range{blocksPerMat * local[0] * yo.info.dims[1],
                              yo.info.dims[2] * yo.info.dims[3] * local[1]};

    // Passing bools to opencl kernels is not allowed
    bool batch =
        !(xo.info.dims[1] == 1 && xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    getQueue().submit([&](sycl::handler &h) {
        auto yoAcc = yo.data->get_access(h);
        auto yiAcc = yi.data->get_access(h);
        auto xoAcc = xo.data->get_access(h);
        sycl::stream debugStream(128, 128, h);

        h.parallel_for(
            sycl::nd_range{global, local},
            approx1Kernel<Ty, Tp>(yoAcc, yo.info, yiAcc, yi.info, xoAcc,
                                  xo.info, xi_beg, Tp(1) / xi_step, (Ty)offGrid,
                                  (int)blocksPerMat, (int)batch, (int)method,
                                  xdim, order));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
