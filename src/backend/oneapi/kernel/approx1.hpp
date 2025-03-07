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
class approx1Kernel {
   public:
    approx1Kernel(write_accessor<Ty> d_yo, const KParam yoInfo,
                  read_accessor<Ty> d_yi, const KParam yiInfo,
                  read_accessor<Tp> d_xo, const KParam xoInfo, const Tp xi_beg,
                  const Tp xi_step_reproc, const Ty offGrid,
                  const int blocksMatX, const af_interp_type method,
                  const bool batch, const int XDIM)
        : d_yo_(d_yo)
        , yoInfo_(yoInfo)
        , d_yi_(d_yi)
        , yiInfo_(yiInfo)
        , d_xo_(d_xo)
        , xoInfo_(xoInfo)
        , xi_beg_(xi_beg)
        , xi_step_reproc_(xi_step_reproc)
        , offGrid_(offGrid)
        , blocksMatX_(blocksMatX)
        , method_(method)
        , batch_(batch)
        , XDIM_(XDIM) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int idw = g.get_group_id(1) / yoInfo_.dims[2];
        const int idz = g.get_group_id(1) - idw * yoInfo_.dims[2];

        const int idy        = g.get_group_id(0) / blocksMatX_;
        const int blockIdx_x = g.get_group_id(0) - idy * blocksMatX_;
        const int idx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);

        if (idx >= yoInfo_.dims[0] || idy >= yoInfo_.dims[1] ||
            idz >= yoInfo_.dims[2] || idw >= yoInfo_.dims[3])
            return;

        // FIXME: Only cubic interpolation is doing clamping
        // We need to make it consistent across all methods
        // Not changing the behavior because tests will fail
        const bool doclamp = order == 3;

        bool is_off[] = {xoInfo_.dims[0] > 1, xoInfo_.dims[1] > 1,
                         xoInfo_.dims[2] > 1, xoInfo_.dims[3] > 1};

        const int yo_idx = idw * yoInfo_.strides[3] + idz * yoInfo_.strides[2] +
                           idy * yoInfo_.strides[1] + idx + yoInfo_.offset;

        int xo_idx = idx * is_off[0] + xoInfo_.offset;
        if (batch_) {
            xo_idx += idw * xoInfo_.strides[3] * is_off[3];
            xo_idx += idz * xoInfo_.strides[2] * is_off[2];
            xo_idx += idy * xoInfo_.strides[1] * is_off[1];
        }

        const Tp x = (d_xo_[xo_idx] - xi_beg_) * xi_step_reproc_;

#pragma unroll
        for (int flagIdx = 0; flagIdx < 4; ++flagIdx) {
            is_off[flagIdx] = true;
        }
        is_off[XDIM_] = false;

        if (x < 0 || yiInfo_.dims[XDIM_] < x + 1) {
            d_yo_[yo_idx] = offGrid_;
            return;
        }

        int yi_idx = idx * is_off[0] + yiInfo_.offset;
        yi_idx += idw * yiInfo_.strides[3] * is_off[3];
        yi_idx += idz * yiInfo_.strides[2] * is_off[2];
        yi_idx += idy * yiInfo_.strides[1] * is_off[1];

        Interp1<Ty, Tp, order> interp;
        interp(d_yo_, yoInfo_, yo_idx, d_yi_, yiInfo_, yi_idx, x, XDIM_,
               method_, 1, doclamp);
    }

   protected:
    write_accessor<Ty> d_yo_;
    const KParam yoInfo_;
    read_accessor<Ty> d_yi_;
    const KParam yiInfo_;
    read_accessor<Tp> d_xo_;
    const KParam xoInfo_;
    const Tp xi_beg_;
    const Tp xi_step_reproc_;
    const Ty offGrid_;
    const int blocksMatX_;
    const af_interp_type method_;
    const bool batch_;
    const int XDIM_;
};

template<typename Ty, typename Tp, int order>
void approx1(Param<Ty> yo, const Param<Ty> yi, const Param<Tp> xo,
             const int xdim, const Tp xi_beg, const Tp xi_step,
             const float offGrid, const af_interp_type method) {
    constexpr int THREADS = 256;

    auto local        = sycl::range{THREADS, 1};
    uint blocksPerMat = divup(yo.info.dims[0], local[0]);
    auto global       = sycl::range{blocksPerMat * local[0] * yo.info.dims[1],
                              yo.info.dims[2] * yo.info.dims[3] * local[1]};

    bool batch =
        !(xo.info.dims[1] == 1 && xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<Ty> yoAcc{*yo.data, h};
        read_accessor<Ty> yiAcc{*yi.data, h};
        read_accessor<Tp> xoAcc{*xo.data, h};

        h.parallel_for(sycl::nd_range{global, local},
                       approx1Kernel<Ty, Tp, order>(
                           yoAcc, yo.info, yiAcc, yi.info, xoAcc, xo.info,
                           xi_beg, Tp(1) / xi_step, (Ty)offGrid,
                           (uint)blocksPerMat, method, batch, xdim));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
