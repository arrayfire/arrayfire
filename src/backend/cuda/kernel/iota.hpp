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
#include <nvrtc_kernel_headers/iota_cuh.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void iota(Param<T> out, const af::dim4 &sdims) {
    constexpr unsigned IOTA_TX = 32;
    constexpr unsigned IOTA_TY = 8;
    constexpr unsigned TILEX   = 512;
    constexpr unsigned TILEY   = 32;

    auto iota = common::getKernel("arrayfire::cuda::iota", {{iota_cuh_src}},
                                  TemplateArgs(TemplateTypename<T>()));

    dim3 threads(IOTA_TX, IOTA_TY, 1);

    int blocksPerMatX = divup(out.dims[0], TILEX);
    int blocksPerMatY = divup(out.dims[1], TILEY);

    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    iota(qArgs, out, sdims[0], sdims[1], sdims[2], sdims[3], blocksPerMatX,
         blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
