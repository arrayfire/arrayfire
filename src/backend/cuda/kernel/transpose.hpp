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
#include <nvrtc_kernel_headers/transpose_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int TILE_DIM  = 32;
static const int THREADS_X = TILE_DIM;
static const int THREADS_Y = 256 / TILE_DIM;

template<typename T>
void transpose(Param<T> out, CParam<T> in, const bool conjugate,
               const bool is32multiple) {
    auto transpose = common::getKernel(
        "arrayfire::cuda::transpose", {{transpose_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(conjugate),
                     TemplateArg(is32multiple)),
        {{DefineValue(TILE_DIM), DefineValue(THREADS_Y)}});

    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], TILE_DIM);
    int blk_y = divup(in.dims[1], TILE_DIM);
    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);
    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    transpose(qArgs, out, in, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
