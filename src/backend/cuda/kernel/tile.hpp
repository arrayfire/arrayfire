/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <nvrtc_kernel_headers/tile_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void tile(Param<T> out, CParam<T> in) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 512;
    constexpr unsigned TILEY = 32;

    auto tile = common::getKernel("arrayfire::cuda::tile", {{tile_cuh_src}},
                                  TemplateArgs(TemplateTypename<T>()));

    dim3 threads(TX, TY, 1);

    int blocksPerMatX = divup(out.dims[0], TILEX);
    int blocksPerMatY = divup(out.dims[1], TILEY);
    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    tile(qArgs, out, in, blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
