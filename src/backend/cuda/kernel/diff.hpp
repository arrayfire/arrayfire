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
#include <nvrtc_kernel_headers/diff_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void diff(Param<T> out, CParam<T> in, const int indims, const unsigned dim,
          const bool isDiff2) {
    constexpr unsigned TX = 16;
    constexpr unsigned TY = 16;

    auto diff =
        common::getKernel("arrayfire::cuda::diff", {{diff_cuh_src}},
                          TemplateArgs(TemplateTypename<T>(), TemplateArg(dim),
                                       TemplateArg(isDiff2)));

    dim3 threads(TX, TY, 1);

    if (dim == 0 && indims == 1) { threads = dim3(TX * TY, 1, 1); }

    int blocksPerMatX = divup(out.dims[0], TX);
    int blocksPerMatY = divup(out.dims[1], TY);
    dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3], 1);

    const int oElem = out.dims[0] * out.dims[1] * out.dims[2] * out.dims[3];

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    diff(qArgs, out, in, oElem, blocksPerMatX, blocksPerMatY);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
