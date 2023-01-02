/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <assign_kernel_param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/assign_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void assign(Param<T> out, CParam<T> in, const AssignKernelParam& p) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    auto assignKer =
        common::getKernel("arrayfire::cuda::assign", {{assign_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));

    const dim3 threads(THREADS_X, THREADS_Y);

    int blks_x = divup(in.dims[0], threads.x);
    int blks_y = divup(in.dims[1], threads.y);

    dim3 blocks(blks_x * in.dims[2], blks_y * in.dims[3]);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    assignKer(qArgs, out, in, p, blks_x, blks_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
