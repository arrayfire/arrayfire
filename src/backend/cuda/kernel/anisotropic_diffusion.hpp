/*******************************************************
 * Copyright (c) 2017, ArrayFire
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
#include <nvrtc_kernel_headers/anisotropic_diffusion_cuh.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cuda {
namespace kernel {

constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int YDIM_LOAD = 2 * THREADS_X / THREADS_Y;

template<typename T>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct,
                          const af::fluxFunction fftype, bool isMCDE) {
    auto diffUpdate = common::getKernel(
        "arrayfire::cuda::diffUpdate", {{anisotropic_diffusion_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(fftype),
                     TemplateArg(isMCDE)),
        {{DefineValue(THREADS_X), DefineValue(THREADS_Y),
          DefineValue(YDIM_LOAD)}});

    dim3 threads(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.dims[0], threads.x);
    int blkY = divup(inout.dims[1], threads.y * YDIM_LOAD);

    dim3 blocks(blkX * inout.dims[2], blkY * inout.dims[3], 1);

    const int maxBlkY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    const int blkZ    = divup(blocks.y, maxBlkY);

    if (blkZ > 1) {
        blocks.y = maxBlkY;
        blocks.z = blkZ;
    }

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    diffUpdate(qArgs, inout, dt, mct, blkX, blkY);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
