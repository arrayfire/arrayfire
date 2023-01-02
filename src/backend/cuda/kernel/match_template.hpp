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
#include <nvrtc_kernel_headers/match_template_cuh.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename inType, typename outType>
void matchTemplate(Param<outType> out, CParam<inType> srch,
                   CParam<inType> tmplt, const af::matchType mType,
                   bool needMean) {
    auto matchTemplate = common::getKernel(
        "arrayfire::cuda::matchTemplate", {{match_template_cuh_src}},
        TemplateArgs(TemplateTypename<inType>(), TemplateTypename<outType>(),
                     TemplateArg(mType), TemplateArg(needMean)));

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(srch.dims[0], threads.x);
    int blk_y = divup(srch.dims[1], threads.y);

    dim3 blocks(blk_x * srch.dims[2], blk_y * srch.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    matchTemplate(qArgs, out, srch, tmplt, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
