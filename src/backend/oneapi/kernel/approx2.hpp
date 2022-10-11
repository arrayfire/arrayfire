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

#define MULRC(a, b) (a) * (b)
#define MULCR(a, b) (a) * (b)

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
class approx2Kernel {
   public:
    approx2Kernel(sycl::accessor<Ty> d_zo, const KParam zo,
                  sycl::accessor<Ty> d_zi, const KParam zi,
                  sycl::accessor<Tp> d_xo, const KParam xo,
                  sycl::accessor<Tp> d_yo, const KParam yo, const Tp xi_beg,
                  const Tp xi_step_reproc, const Tp yi_beg,
                  const Tp yi_step_reproc, const Ty offGrid,
                  const int blocksMatX, const int blocksMatY, const int batch,
                  int method, const int XDIM, const int YDIM,
                  const int INTERP_ORDER)
        : d_zo_(d_zo)
        , zo_(zo)
        , d_zi_(d_zi)
        , zi_(zi)
        , d_xo_(d_xo)
        , xo_(xo)
        , d_yo_(d_yo)
        , yo_(yo)
        , xi_beg_(xi_beg)
        , xi_step_reproc_(xi_step_reproc)
        , yi_beg_(yi_beg)
        , yi_step_reproc_(yi_step_reproc)
        , offGrid_(offGrid)
        , blocksMatX_(blocksMatX)
        , blocksMatY_(blocksMatY)
        , batch_(batch)
        , method_(method)
        , XDIM_(XDIM)
        , YDIM_(YDIM)
        , INTERP_ORDER_(INTERP_ORDER) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int idz = g.get_group_id(0) / blocksMatX_;
        const int idw = g.get_group_id(1) / blocksMatY_;

        const int blockIdx_x = g.get_group_id(0) - idz * blocksMatX_;
        const int blockIdx_y = g.get_group_id(1) - idw * blocksMatY_;

        const int idx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int idy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        if (idx >= zo_.dims[0] || idy >= zo_.dims[1] || idz >= zo_.dims[2] ||
            idw >= zo_.dims[3])
            return;

        // FIXME: Only cubic interpolation is doing clamping
        // We need to make it consistent across all methods
        // Not changing the behavior because tests will fail
        const bool doclamp = INTERP_ORDER_ == 3;

        bool is_off[] = {xo_.dims[0] > 1, xo_.dims[1] > 1, xo_.dims[2] > 1,
                         xo_.dims[3] > 1};

        const int zo_idx = idw * zo_.strides[3] + idz * zo_.strides[2] +
                           idy * zo_.strides[1] + idx + zo_.offset;
        int xo_idx =
            idy * xo_.strides[1] * is_off[1] + idx * is_off[0] + xo_.offset;

        int yo_idx =
            idy * yo_.strides[1] * is_off[1] + idx * is_off[0] + yo_.offset;
        if (batch_) {
            xo_idx += idw * xo_.strides[3] * is_off[3] +
                      idz * xo_.strides[2] * is_off[2];
            yo_idx += idw * yo_.strides[3] * is_off[3] +
                      idz * yo_.strides[2] * is_off[2];
        }

#pragma unroll
        for (int flagIdx = 0; flagIdx < 4; ++flagIdx) {
            is_off[flagIdx] = true;
        }
        is_off[XDIM_] = false;
        is_off[YDIM_] = false;

        const Tp x = (d_xo_[xo_idx] - xi_beg_) * xi_step_reproc_;
        const Tp y = (d_yo_[yo_idx] - yi_beg_) * yi_step_reproc_;

        if (x < 0 || y < 0 || zi_.dims[XDIM_] < x + 1 ||
            zi_.dims[YDIM_] < y + 1) {
            d_zo_[zo_idx] = offGrid_;
            return;
        }

        int zi_idx =
            idy * zi_.strides[1] * is_off[1] + idx * is_off[0] + zi_.offset;
        zi_idx +=
            idw * zi_.strides[3] * is_off[3] + idz * zi_.strides[2] * is_off[2];

        if (INTERP_ORDER_ == 1)
            interp2o1(d_zo_, zo_, zo_idx, d_zi_, zi_, zi_idx, x, y, method_, 1,
                      doclamp, 2);
        if (INTERP_ORDER_ == 2)
            interp2o2(d_zo_, zo_, zo_idx, d_zi_, zi_, zi_idx, x, y, method_, 1,
                      doclamp, 2);
        if (INTERP_ORDER_ == 3)
            interp2o3(d_zo_, zo_, zo_idx, d_zi_, zi_, zi_idx, x, y, method_, 1,
                      doclamp, 2);
    }

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

    void interp2o1(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x, Tp y,
                   int method, int batch, bool doclamp, int batch_dim) const {
        int xid = (method == AF_INTERP_LOWER ? floor(x) : round(x));
        int yid = (method == AF_INTERP_LOWER ? floor(y) : round(y));

        const int x_lim    = in.dims[XDIM_];
        const int y_lim    = in.dims[YDIM_];
        const int x_stride = in.strides[XDIM_];
        const int y_stride = in.strides[YDIM_];

        if (doclamp) {
            xid = fmax(0, fmin(xid, x_lim));
            yid = fmax(0, fmin(yid, y_lim));
        }
        const int idx = ioff + yid * y_stride + xid * x_stride;

        bool condX = xid >= 0 && xid < x_lim;
        bool condY = yid >= 0 && yid < y_lim;

        Ty zero = (Ty)0;
        ;
        bool cond = condX && condY;
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            d_out[ooff + n * out.strides[batch_dim]] =
                (doclamp || cond) ? d_in[idx_n] : zero;
        }
    }

    void interp2o2(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x, Tp y,
                   int method, int batch, bool doclamp, int batch_dim) const {
        const int grid_x = floor(x);
        const Tp off_x   = x - grid_x;

        const int grid_y = floor(y);
        const Tp off_y   = y - grid_y;

        const int x_lim    = in.dims[XDIM_];
        const int y_lim    = in.dims[YDIM_];
        const int x_stride = in.strides[XDIM_];
        const int y_stride = in.strides[YDIM_];
        const int idx      = ioff + grid_y * y_stride + grid_x * x_stride;

        bool condX[2] = {true, x + 1 < x_lim};
        bool condY[2] = {true, y + 1 < y_lim};
        int offx[2]   = {0, condX[1] ? 1 : 0};
        int offy[2]   = {0, condY[1] ? 1 : 0};

        Tp xratio = off_x, yratio = off_y;
        if (method == AF_INTERP_LINEAR_COSINE) {
            xratio = (1 - cos(xratio * (Tp)M_PI)) / 2;
            yratio = (1 - cos(yratio * (Tp)M_PI)) / 2;
        }

        Ty zero = (Ty)0;
        ;
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            Ty val[2][2];
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

    void interp2o3(sycl::accessor<Ty> d_out, KParam out, int ooff,
                   sycl::accessor<Ty> d_in, KParam in, int ioff, Tp x, Tp y,
                   int method, int batch, bool doclamp, int batch_dim) const {
        const int grid_x = floor(x);
        const Tp off_x   = x - grid_x;

        const int grid_y = floor(y);
        const Tp off_y   = y - grid_y;

        const int x_lim    = in.dims[XDIM_];
        const int y_lim    = in.dims[YDIM_];
        const int x_stride = in.strides[XDIM_];
        const int y_stride = in.strides[YDIM_];
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

        Ty zero = (Ty)0;
        ;
        for (int n = 0; n < batch; n++) {
            int idx_n = idx + n * in.strides[batch_dim];
            // for bicubic interpolation, work with 4x4 val at a time
            Ty val[4][4];
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

   private:
    sycl::accessor<Ty> d_zo_;
    const KParam zo_;
    sycl::accessor<Ty> d_zi_;
    const KParam zi_;
    sycl::accessor<Tp> d_xo_;
    const KParam xo_;
    sycl::accessor<Tp> d_yo_;
    const KParam yo_;
    const Tp xi_beg_;
    const Tp xi_step_reproc_;
    const Tp yi_beg_;
    const Tp yi_step_reproc_;
    const Ty offGrid_;
    const int blocksMatX_;
    const int blocksMatY_;
    const int batch_;
    int method_;
    const int XDIM_;
    const int YDIM_;
    const int INTERP_ORDER_;
};

template<typename Ty, typename Tp>
void approx2(Param<Ty> zo, const Param<Ty> zi, const Param<Tp> xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Param<Tp> yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const float offGrid,
             const af_interp_type method, const int order) {
    constexpr int TX = 16;
    constexpr int TY = 16;

    auto local          = sycl::range{TX, TY};
    dim_t blocksPerMatX = divup(zo.info.dims[0], local[0]);
    dim_t blocksPerMatY = divup(zo.info.dims[1], local[1]);
    auto global = sycl::range{blocksPerMatX * local[0] * zo.info.dims[2],
                              blocksPerMatY * local[1] * zo.info.dims[3]};

    // Passing bools to opencl kernels is not allowed
    bool batch = !(xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    getQueue().submit([&](sycl::handler &h) {
        auto zoAcc = zo.data->get_access(h);
        auto ziAcc = zi.data->get_access(h);
        auto xoAcc = xo.data->get_access(h);
        auto yoAcc = yo.data->get_access(h);
        sycl::stream debugStream(128, 128, h);

        h.parallel_for(
            sycl::nd_range{global, local},
            approx2Kernel<Ty, Tp>(
                zoAcc, zo.info, ziAcc, zi.info, xoAcc, xo.info, yoAcc, yo.info,
                xi_beg, Tp(1) / xi_step, yi_beg, Tp(1) / yi_step, (Ty)offGrid,
                static_cast<int>(blocksPerMatX),
                static_cast<int>(blocksPerMatY), static_cast<int>(batch),
                static_cast<int>(method), xdim, ydim, order));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
