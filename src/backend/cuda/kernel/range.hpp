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
#include <nvrtc_kernel_headers/range_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void range(Param<T> out, const int dim) {
    constexpr unsigned RANGE_TX    = 32;
    constexpr unsigned RANGE_TY    = 8;
    constexpr unsigned RANGE_TILEX = 512;
    constexpr unsigned RANGE_TILEY = 32;

    auto range = common::getKernel("arrayfire::cuda::range", {{range_cuh_src}},
                                   TemplateArgs(TemplateTypename<T>()));

    dim3 threads(RANGE_TX, RANGE_TY, 1);

    int blocksPerMatX = divup(out.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.dims[1], RANGE_TILEY);
    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    range(qArgs, out, dim, blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
