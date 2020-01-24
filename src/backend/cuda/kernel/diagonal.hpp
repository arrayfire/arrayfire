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
#include <nvrtc_kernel_headers/diagonal_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

template<typename T>
void diagCreate(Param<T> out, CParam<T> in, int num) {
    static const std::string src(diagonal_cuh, diagonal_cuh_len);

    auto genDiagMat = common::findKernel("cuda::createDiagonalMat", {src},
                                         {TemplateTypename<T>()});

    dim3 threads(32, 8);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);
    dim3 blocks(blocks_x * out.dims[2], blocks_y);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    const int blocksPerMatZ = divup(blocks.y, maxBlocksY);
    if (blocksPerMatZ > 1) {
        blocks.y = maxBlocksY;
        blocks.z = blocksPerMatZ;
    }

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    genDiagMat(qArgs, out, in, num, blocks_x);

    POST_LAUNCH_CHECK();
}

template<typename T>
void diagExtract(Param<T> out, CParam<T> in, int num) {
    static const std::string src(diagonal_cuh, diagonal_cuh_len);

    auto extractDiag = common::findKernel("cuda::extractDiagonal", {src},
                                          {TemplateTypename<T>()});

    dim3 threads(256, 1);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_z = out.dims[2];
    dim3 blocks(blocks_x, out.dims[3] * blocks_z);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    extractDiag(qArgs, out, in, num, blocks_z);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
