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
#include <kernel/accessors.hpp>
#include <kernel/interp.hpp>
#include <traits.hpp>
#include <af/constants.h>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = 256 / TILE_DIM;

template<typename Ty, typename Tp, int order>
class approx2Kernel {
   public:
    approx2Kernel(write_accessor<Ty> d_zo, const KParam zo,
                  read_accessor<Ty> d_zi, const KParam zi,
                  read_accessor<Tp> d_xo, const KParam xo,
                  read_accessor<Tp> d_yo, const KParam yo, const Tp xi_beg,
                  const Tp xi_step_reproc, const Tp yi_beg,
                  const Tp yi_step_reproc, const Ty offGrid,
                  const int blocksMatX, const int blocksMatY, const bool batch,
                  const af_interp_type method, const int XDIM, const int YDIM)
        : d_zo_(d_zo)
        , zoInfo_(zo)
        , d_zi_(d_zi)
        , ziInfo_(zi)
        , d_xo_(d_xo)
        , xoInfo_(xo)
        , d_yo_(d_yo)
        , yoInfo_(yo)
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
        , YDIM_(YDIM) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int idz = g.get_group_id(0) / blocksMatX_;
        const int idw = g.get_group_id(1) / blocksMatY_;

        const int blockIdx_x = g.get_group_id(0) - idz * blocksMatX_;
        const int blockIdx_y = g.get_group_id(1) - idw * blocksMatY_;

        const int idx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int idy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        if (idx >= zoInfo_.dims[0] || idy >= zoInfo_.dims[1] ||
            idz >= zoInfo_.dims[2] || idw >= zoInfo_.dims[3])
            return;

        // FIXME: Only cubic interpolation is doing clamping
        // We need to make it consistent across all methods
        // Not changing the behavior because tests will fail
        const bool doclamp = order == 3;

        bool is_off[] = {xoInfo_.dims[0] > 1, xoInfo_.dims[1] > 1,
                         xoInfo_.dims[2] > 1, xoInfo_.dims[3] > 1};

        const int zo_idx = idw * zoInfo_.strides[3] + idz * zoInfo_.strides[2] +
                           idy * zoInfo_.strides[1] + idx + zoInfo_.offset;
        int xo_idx = idy * xoInfo_.strides[1] * is_off[1] + idx * is_off[0] +
                     xoInfo_.offset;

        int yo_idx = idy * yoInfo_.strides[1] * is_off[1] + idx * is_off[0] +
                     yoInfo_.offset;
        if (batch_) {
            xo_idx += idw * xoInfo_.strides[3] * is_off[3] +
                      idz * xoInfo_.strides[2] * is_off[2];
            yo_idx += idw * yoInfo_.strides[3] * is_off[3] +
                      idz * yoInfo_.strides[2] * is_off[2];
        }

#pragma unroll
        for (int flagIdx = 0; flagIdx < 4; ++flagIdx) {
            is_off[flagIdx] = true;
        }
        is_off[XDIM_] = false;
        is_off[YDIM_] = false;

        const Tp x = (d_xo_[xo_idx] - xi_beg_) * xi_step_reproc_;
        const Tp y = (d_yo_[yo_idx] - yi_beg_) * yi_step_reproc_;

        if (x < 0 || y < 0 || ziInfo_.dims[XDIM_] < x + 1 ||
            ziInfo_.dims[YDIM_] < y + 1) {
            d_zo_[zo_idx] = offGrid_;
            return;
        }

        int zi_idx = idy * ziInfo_.strides[1] * is_off[1] + idx * is_off[0] +
                     ziInfo_.offset;
        zi_idx += idw * ziInfo_.strides[3] * is_off[3] +
                  idz * ziInfo_.strides[2] * is_off[2];

        Interp2<Ty, Tp, order> interp;
        interp(d_zo_, zoInfo_, zo_idx, d_zi_, ziInfo_, zi_idx, x, y, XDIM_,
               YDIM_, method_, 1, doclamp);
    }

   protected:
    write_accessor<Ty> d_zo_;
    const KParam zoInfo_;
    read_accessor<Ty> d_zi_;
    const KParam ziInfo_;
    read_accessor<Tp> d_xo_;
    const KParam xoInfo_;
    read_accessor<Tp> d_yo_;
    const KParam yoInfo_;
    const Tp xi_beg_;
    const Tp xi_step_reproc_;
    const Tp yi_beg_;
    const Tp yi_step_reproc_;
    const Ty offGrid_;
    const int blocksMatX_;
    const int blocksMatY_;
    const int batch_;
    af::interpType method_;
    const int XDIM_;
    const int YDIM_;
};

template<typename Ty, typename Tp, int order>
void approx2(Param<Ty> zo, const Param<Ty> zi, const Param<Tp> xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Param<Tp> yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const float offGrid,
             const af_interp_type method) {
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
        write_accessor<Ty> zoAcc{*zo.data, h};
        read_accessor<Ty> ziAcc{*zi.data, h};
        read_accessor<Tp> xoAcc{*xo.data, h};
        read_accessor<Tp> yoAcc{*yo.data, h};

        h.parallel_for(
            sycl::nd_range{global, local},
            approx2Kernel<Ty, Tp, order>(
                zoAcc, zo.info, ziAcc, zi.info, xoAcc, xo.info, yoAcc, yo.info,
                xi_beg, Tp(1) / xi_step, yi_beg, Tp(1) / yi_step, (Ty)offGrid,
                static_cast<int>(blocksPerMatX),
                static_cast<int>(blocksPerMatY), batch, method, xdim, ydim));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
