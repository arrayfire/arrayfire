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
#include <debug_cuda.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/lookup_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int THRD_LOAD = THREADS_X / THREADS_Y;

template<typename in_t, typename idx_t>
void lookup(Param<in_t> out, CParam<in_t> in, CParam<idx_t> indices, int nDims,
            unsigned dim) {
    static const std::string src(lookup_cuh, lookup_cuh_len);

    /* find which dimension has non-zero # of elements */
    unsigned vDim = 0;
    for (int i = 0; i < 4; i++) {
        if (in.dims[i] == 1)
            vDim++;
        else
            break;
    }

    if (dim == 0 && nDims == 1 && dim == vDim) {
        const dim3 threads(THREADS, 1);

        int blks = divup(out.dims[vDim], THREADS * THRD_LOAD);

        dim3 blocks(blks, 1);

        auto lookup1d =
            getKernel("cuda::lookup1D", src,
                      {TemplateTypename<in_t>(), TemplateTypename<idx_t>()},
                      {DefineValue(THREADS), DefineValue(THRD_LOAD)});

        EnqueueArgs qArgs(blocks, threads, getActiveStream());

        lookup1d(qArgs, out, in, indices, vDim);
    } else {
        const dim3 threads(THREADS_X, THREADS_Y);

        int blks_x = divup(out.dims[0], threads.x);
        int blks_y = divup(out.dims[1], threads.y);

        dim3 blocks(blks_x * out.dims[2], blks_y * out.dims[3]);

        const int maxBlocksY =
            cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
        blocks.z = divup(blocks.y, maxBlocksY);
        blocks.y = divup(blocks.y, blocks.z);

        auto lookupnd =
            getKernel("cuda::lookupND", src,
                      {TemplateTypename<in_t>(), TemplateTypename<idx_t>(),
                       TemplateArg(dim)});
        EnqueueArgs qArgs(blocks, threads, getActiveStream());

        lookupnd(qArgs, out, in, indices, blks_x, blks_y);
    }
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
