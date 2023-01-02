/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <math.hpp>
#include <nvrtc_kernel_headers/select_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

constexpr uint DIMX  = 32;
constexpr uint DIMY  = 8;
constexpr int REPEAT = 64;

template<typename T>
void select(Param<T> out, CParam<char> cond, CParam<T> a, CParam<T> b,
            int ndims) {
    bool is_same = true;
    for (int i = 0; i < 4; i++) { is_same &= (a.dims[i] == b.dims[i]); }

    auto select = common::getKernel(
        "arrayfire::cuda::select", {{select_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(is_same)));

    dim3 threads(DIMX, DIMY);

    if (ndims == 1) {
        threads.x *= threads.y;
        threads.y = 1;
    }

    int blk_x = divup(out.dims[0], REPEAT * threads.x);
    int blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x * out.dims[2], blk_y * out.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    select(qArgs, out, cond, a, b, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

template<typename T>
void select_scalar(Param<T> out, CParam<char> cond, CParam<T> a, const double b,
                   int ndims, bool flip) {
    auto selectScalar = common::getKernel(
        "arrayfire::cuda::selectScalar", {{select_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(flip)));

    dim3 threads(DIMX, DIMY);

    if (ndims == 1) {
        threads.x *= threads.y;
        threads.y = 1;
    }

    int blk_x = divup(out.dims[0], REPEAT * threads.x);
    int blk_y = divup(out.dims[1], threads.y);

    dim3 blocks(blk_x * out.dims[2], blk_y * out.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    selectScalar(qArgs, out, cond, a, scalar<T>(b), blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
