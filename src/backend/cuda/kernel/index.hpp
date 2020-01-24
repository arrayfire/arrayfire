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
#include <assign_kernel_param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/index_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void index(Param<T> out, CParam<T> in, const IndexKernelParam& p) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    static const std::string source(index_cuh, index_cuh_len);

    auto index =
        common::findKernel("cuda::index", {source}, {TemplateTypename<T>()});

    const dim3 threads(THREADS_X, THREADS_Y);

    int blks_x = divup(out.dims[0], threads.x);
    int blks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blks_x * out.dims[2], blks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    index(qArgs, out, in, p, blks_x, blks_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
