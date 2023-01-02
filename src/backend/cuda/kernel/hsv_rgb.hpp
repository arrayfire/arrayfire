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
#include <nvrtc_kernel_headers/hsv_rgb_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T>
void hsv2rgb_convert(Param<T> out, CParam<T> in, bool isHSV2RGB) {
    auto hsvrgbConverter = common::getKernel(
        "arrayfire::cuda::hsvrgbConverter", {{hsv_rgb_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(isHSV2RGB)));

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    // all images are three channels, so batch
    // parameter would be along 4th dimension
    dim3 blocks(blk_x * in.dims[3], blk_y);

    const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
    blocks.z             = divup(blocks.y, maxBlocksY);
    blocks.y             = divup(blocks.y, blocks.z);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    hsvrgbConverter(qArgs, out, in, blk_x);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
