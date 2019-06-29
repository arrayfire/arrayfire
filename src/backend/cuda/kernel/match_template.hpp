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
#include <debug_cuda.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/match_template_cuh.hpp>
#include <af/defines.h>

#include <string>

namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename inType, typename outType>
void matchTemplate(Param<outType> out, CParam<inType> srch,
                   CParam<inType> tmplt, const af::matchType mType,
                   bool needMean) {
    static const std::string source(match_template_cuh, match_template_cuh_len);

    auto match = getKernel("cuda::matchTemplate", source,
            {
              TemplateTypename<inType>(),
              TemplateTypename<outType>(),
              TemplateArg(mType),
              TemplateArg(needMean)
            }
            );

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(srch.dims[0], threads.x);
    int blk_y = divup(srch.dims[1], threads.y);

    dim3 blocks(blk_x * srch.dims[2], blk_y * srch.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    match(qArgs, out, srch, tmplt, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
