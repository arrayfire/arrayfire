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

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void identity(Param<T> out) {
    static const std::string source(identity_cuh, identity_cuh_len);

    auto identity =
        common::getKernel("cuda::identity", {source}, {TemplateTypename<T>()});

    dim3 threads(32, 8);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);
    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    identity(qArgs, out, blocks_x, blocks_y);
    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
