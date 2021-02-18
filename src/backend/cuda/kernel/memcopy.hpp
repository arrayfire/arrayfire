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
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <dims_param.hpp>
#include <nvrtc_kernel_headers/copy_cuh.hpp>
#include <nvrtc_kernel_headers/memcopy_cuh.hpp>

#include <algorithm>

namespace cuda {
namespace kernel {

constexpr uint DIMX = 32;
constexpr uint DIMY = 8;

template<typename T>
void memcopy(Param<T> out, CParam<T> in, const dim_t ndims) {
    auto memCopy = common::getKernel("cuda::memcopy", {memcopy_cuh_src},
                                     {TemplateTypename<T>()});

    dim3 threads(DIMX, DIMY);

    if (ndims == 1) {
        threads.x *= threads.y;
        threads.y = 1;
    }

    // FIXME: DO more work per block
    uint blocks_x = divup(in.dims[0], threads.x);
    uint blocks_y = divup(in.dims[1], threads.y);

    dim3 blocks(blocks_x * in.dims[2], blocks_y * in.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    memCopy(qArgs, out, in, blocks_x, blocks_y);

    POST_LAUNCH_CHECK();
}

template<typename inType, typename outType>
void copy(Param<outType> dst, CParam<inType> src, int ndims,
          outType default_value, double factor) {
    dim3 threads(DIMX, DIMY);
    size_t local_size[] = {DIMX, DIMY};

    // FIXME: Why isn't threads being updated??
    local_size[0] *= local_size[1];
    if (ndims == 1) { local_size[1] = 1; }

    uint blk_x = divup(dst.dims[0], local_size[0]);
    uint blk_y = divup(dst.dims[1], local_size[1]);

    dim3 blocks(blk_x * dst.dims[2], blk_y * dst.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    int trgt_l       = std::min(dst.dims[3], src.dims[3]);
    int trgt_k       = std::min(dst.dims[2], src.dims[2]);
    int trgt_j       = std::min(dst.dims[1], src.dims[1]);
    int trgt_i       = std::min(dst.dims[0], src.dims[0]);
    dims_t trgt_dims = {{trgt_i, trgt_j, trgt_k, trgt_l}};

    bool same_dims =
        ((src.dims[0] == dst.dims[0]) && (src.dims[1] == dst.dims[1]) &&
         (src.dims[2] == dst.dims[2]) && (src.dims[3] == dst.dims[3]));

    auto copy = common::getKernel(
        "cuda::copy", {copy_cuh_src},
        {TemplateTypename<inType>(), TemplateTypename<outType>(),
         TemplateArg(same_dims)});

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    copy(qArgs, dst, src, default_value, factor, trgt_dims, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
