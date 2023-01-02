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
#include <nvrtc_kernel_headers/sobel_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename Ti, typename To>
void sobel(Param<To> dx, Param<To> dy, CParam<Ti> in,
           const unsigned& ker_size) {
    UNUSED(ker_size);

    auto sobel3x3 = common::getKernel(
        "arrayfire::cuda::sobel3x3", {{sobel_cuh_src}},
        TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<To>()),
        {{DefineValue(THREADS_X), DefineValue(THREADS_Y)}});

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    // TODO: call other cases when support for 5x5 & 7x7 is added
    // Note: This is checked at sobel API entry point
    sobel3x3(qArgs, dx, dy, in, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
