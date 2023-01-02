/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <kernel/config.hpp>
#include <nvrtc_kernel_headers/wrap_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void wrap(Param<T> out, CParam<T> in, const int wx, const int wy, const int sx,
          const int sy, const int px, const int py, const bool is_column) {
    auto wrap = common::getKernel(
        "arrayfire::cuda::wrap", {{wrap_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(is_column)));

    int nx = (out.dims[0] + 2 * px - wx) / sx + 1;
    int ny = (out.dims[1] + 2 * py - wy) / sy + 1;

    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    wrap(qArgs, out, in, wx, wy, sx, sy, px, py, nx, ny, blocks_x, blocks_y);
    POST_LAUNCH_CHECK();
}

template<typename T>
void wrap_dilated(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy, const dim_t px,
                  const dim_t py, const dim_t dx, const dim_t dy,
                  const bool is_column) {
    auto wrap = common::getKernel(
        "arrayfire::cuda::wrap_dilated", {{wrap_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(is_column)));

    int nx = 1 + (out.dims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    int ny = 1 + (out.dims[1] + 2 * py - (((wy - 1) * dy) + 1)) / sy;

    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    wrap(qArgs, out, in, wx, wy, sx, sy, px, py, dx, dy, nx, ny, blocks_x,
         blocks_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
