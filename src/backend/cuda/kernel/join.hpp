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
#include <nvrtc_kernel_headers/join_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void join(Param<T> out, CParam<T> X, const af::dim4 &offset, int dim) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 256;
    constexpr unsigned TILEY = 32;

    static const std::string source(join_cuh, join_cuh_len);

    auto join =
        common::findKernel("cuda::join", {source}, {TemplateTypename<T>()});

    dim3 threads(TX, TY, 1);

    int blocksPerMatX = divup(X.dims[0], TILEX);
    int blocksPerMatY = divup(X.dims[1], TILEY);

    dim3 blocks(blocksPerMatX * X.dims[2], blocksPerMatY * X.dims[3], 1);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    join(qArgs, out, X, offset[0], offset[1], offset[2], offset[3],
         blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
