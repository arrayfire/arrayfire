/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
#include <nvrtc_kernel_headers/pad_array_borders_cuh.hpp>
#include <af/defines.h>

#include <array>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int PADB_THREADS_X = 32;
static const int PADB_THREADS_Y = 8;

template<typename T>
void padBorders(Param<T> out, CParam<T> in, dim4 const lBoundPadding,
                const af::borderType btype) {
    auto padBorders = common::getKernel(
        "arrayfire::cuda::padBorders", {{pad_array_borders_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(btype)));

    dim3 threads(kernel::PADB_THREADS_X, kernel::PADB_THREADS_Y);

    int blk_x = divup(out.dims[0], PADB_THREADS_X);
    int blk_y = divup(out.dims[1], PADB_THREADS_Y);

    dim3 blocks(blk_x * out.dims[2], blk_y * out.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    padBorders(qArgs, out, in, lBoundPadding[0], lBoundPadding[1],
               lBoundPadding[2], lBoundPadding[3], blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
