/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/approx1_cuh.hpp>
#include <nvrtc_kernel_headers/approx2_cuh.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const int TX      = 16;
static const int TY      = 16;
static const int THREADS = 256;

template<typename Ty, typename Tp>
void approx1(Param<Ty> yo, CParam<Ty> yi, CParam<Tp> xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, const float offGrid,
             const af::interpType method, const int order) {
    auto approx1 = common::getKernel(
        "arrayfire::cuda::approx1", {{approx1_cuh_src}},
        TemplateArgs(TemplateTypename<Ty>(), TemplateTypename<Tp>(),
                     TemplateArg(xdim), TemplateArg(order)));

    dim3 threads(THREADS, 1, 1);
    int blocksPerMat = divup(yo.dims[0], threads.x);
    dim3 blocks(blocksPerMat * yo.dims[1], yo.dims[2] * yo.dims[3]);

    bool batch = !(xo.dims[1] == 1 && xo.dims[2] == 1 && xo.dims[3] == 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    approx1(qArgs, yo, yi, xo, xi_beg, Tp(1) / xi_step, offGrid, blocksPerMat,
            batch, method);

    POST_LAUNCH_CHECK();
}

template<typename Ty, typename Tp>
void approx2(Param<Ty> zo, CParam<Ty> zi, CParam<Tp> xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, CParam<Tp> yo, const int ydim,
             const Tp &yi_beg, const Tp &yi_step, const float offGrid,
             const af::interpType method, const int order) {
    auto approx2 = common::getKernel(
        "arrayfire::cuda::approx2", {{approx2_cuh_src}},
        TemplateArgs(TemplateTypename<Ty>(), TemplateTypename<Tp>(),
                     TemplateArg(xdim), TemplateArg(ydim), TemplateArg(order)));

    dim3 threads(TX, TY, 1);
    int blocksPerMatX = divup(zo.dims[0], threads.x);
    int blocksPerMatY = divup(zo.dims[1], threads.y);
    dim3 blocks(blocksPerMatX * zo.dims[2], blocksPerMatY * zo.dims[3]);

    bool batch = !(xo.dims[2] == 1 && xo.dims[3] == 1);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    approx2(qArgs, zo, zi, xo, xi_beg, Tp(1) / xi_step, yo, yi_beg,
            Tp(1) / yi_step, offGrid, blocksPerMatX, blocksPerMatY, batch,
            method);

    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
