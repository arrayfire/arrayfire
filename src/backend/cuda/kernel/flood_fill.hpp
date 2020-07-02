/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/defines.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/flood_fill_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = THREADS / TILE_DIM;

// Shared memory per block required by floodFill kernel
template<typename T>
constexpr size_t sharedMemRequiredByFloodFill() {
    // 1-pixel border neighborhood
    return sizeof(T) * ((THREADS_X + 2) * (THREADS_Y + 2));
}

template<typename T>
void floodFill(Param<T> out, CParam<T> image, CParam<uint> seedsx,
               CParam<uint> seedsy, const T newValue, const T lowValue,
               const T highValue, const af::connectivity nlookup) {
    UNUSED(nlookup);
    static const std::string source(flood_fill_cuh, flood_fill_cuh_len);

    if (sharedMemRequiredByFloodFill<T>() >
        cuda::getDeviceProp(cuda::getActiveDeviceId()).sharedMemPerBlock) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCurrent thread's CUDA device doesn't have sufficient "
                 "shared memory required by FloodFill\n");
        CUDA_NOT_SUPPORTED(errMessage);
    }

    auto initSeeds =
        common::getKernel("cuda::initSeeds", {source}, {TemplateTypename<T>()});
    auto floodStep =
        common::getKernel("cuda::floodStep", {source}, {TemplateTypename<T>()},
                          {DefineValue(THREADS_X), DefineValue(THREADS_Y)});
    auto finalizeOutput = common::getKernel("cuda::finalizeOutput", {source},
                                            {TemplateTypename<T>()});

    EnqueueArgs qArgs(dim3(divup(seedsx.elements(), THREADS)), dim3(THREADS),
                      getActiveStream());
    initSeeds(qArgs, out, seedsx, seedsy);
    POST_LAUNCH_CHECK();

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks(divup(image.dims[0], threads.x),
                divup(image.dims[1], threads.y));
    EnqueueArgs fQArgs(blocks, threads, getActiveStream());

    auto continueFlagPtr = floodStep.getDevPtr("doAnotherLaunch");

    for (int doAnotherLaunch = 1; doAnotherLaunch > 0;) {
        doAnotherLaunch = 0;
        floodStep.setScalar(continueFlagPtr, &doAnotherLaunch);
        floodStep(fQArgs, out, image, lowValue, highValue);
        POST_LAUNCH_CHECK();
        doAnotherLaunch = floodStep.getScalar(continueFlagPtr);
    }
    finalizeOutput(fQArgs, out, newValue);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
