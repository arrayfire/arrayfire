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
#include <nvrtc_kernel_headers/identity_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void identity(Param<T> out) {
    auto identity =
        common::getKernel("arrayfire::cuda::identity", {{identity_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));

    dim3 threads(32, 8);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    identity(qArgs, out, blocks_x, blocks_y);
    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
