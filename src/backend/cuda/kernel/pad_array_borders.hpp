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
#include <debug_cuda.hpp>
#include <internal_enums.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/pad_array_borders.hpp>

#include <string>

namespace cuda {
namespace kernel {

static const int PADB_THREADS_X = 32;
static const int PADB_THREADS_Y = 8;

template<typename T>
void padBorders(Param<T> out, CParam<T> in, dim4 const lBoundPadding,
                const BorderType btype) {
    static const std::string source(pad_array_borders_cuh,
                                    pad_array_borders_cuh_len);
    // clang-format off
    auto padBorders = getKernel("cuda::padBorders", source,
            {
              TemplateTypename<T>(),
              TemplateArg(btype)
            }
            );
    // clang-format on

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
