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
#include <debug_cuda.hpp>
#include <kernel/config.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/wrap_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void wrap(Param<T> out, CParam<T> in, const int wx, const int wy, const int sx,
          const int sy, const int px, const int py, const bool is_column) {
    static const std::string source(wrap_cuh, wrap_cuh_len);

    auto wrap = getKernel("cuda::wrap", source,
                          {TemplateTypename<T>(), TemplateArg(is_column)});

    int nx = (out.dims[0] + 2 * px - wx) / sx + 1;
    int ny = (out.dims[1] + 2 * py - wy) / sy + 1;

    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    wrap(qArgs, out, in, wx, wy, sx, sy, px, py, nx, ny, blocks_x, blocks_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
