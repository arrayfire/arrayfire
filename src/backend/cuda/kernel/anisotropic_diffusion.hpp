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
#include <debug_cuda.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/anisotropic_diffusion_cuh.hpp>
#include <af/defines.h>

#include <string>

namespace cuda {
namespace kernel {

static const int THREADS_X = 32;
static const int THREADS_Y = 8;

template<typename T>
void anisotropicDiffusion(Param<T> inout, const float dt, const float mct,
                          const af::fluxFunction fftype, bool isMCDE) {
    static const std::string source(anisotropic_diffusion_cuh,
                                    anisotropic_diffusion_cuh_len);
    auto diffuse = getKernel("cuda::diffUpdate", source,
            {
              TemplateTypename<T>(),
              TemplateArg(isMCDE)
            },
            {
              DefineValue(THREADS_X),
              DefineValue(THREADS_Y)
            }
            );

    dim3 threads(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.dims[0], threads.x);
    int blkY = divup(inout.dims[1], threads.y);

    dim3 blocks(blkX * inout.dims[2], blkY * inout.dims[3], 1);

    const int maxBlkY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    const int blkZ = divup(blocks.y, maxBlkY);

    if (blkZ > 1) {
        blocks.y = maxBlkY;
        blocks.z = blkZ;
    }

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    diffuse(qArgs, inout, dt, mct, fftype, blkX, blkY);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
