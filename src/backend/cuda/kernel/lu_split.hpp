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
#include <nvrtc_kernel_headers/lu_split_cuh.hpp>

#include <array>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void lu_split(Param<T> lower, Param<T> upper, Param<T> in) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 128;
    constexpr unsigned TILEY = 32;

    const bool sameDims =
        lower.dims[0] == in.dims[0] && lower.dims[1] == in.dims[1];

    auto luSplit = common::getKernel(
        "arrayfire::cuda::luSplit", {{lu_split_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(sameDims)));

    dim3 threads(TX, TY, 1);

    int blocksPerMatX = divup(in.dims[0], TILEX);
    int blocksPerMatY = divup(in.dims[1], TILEY);
    dim3 blocks(blocksPerMatX * in.dims[2], blocksPerMatY * in.dims[3], 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    luSplit(qArgs, lower, upper, in, blocksPerMatX, blocksPerMatY);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
