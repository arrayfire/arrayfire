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
#include <nvrtc_kernel_headers/resize_cuh.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const unsigned TX = 16;
static const unsigned TY = 16;

template<typename T>
void resize(Param<T> out, CParam<T> in, af_interp_type method) {
    auto resize = common::getKernel(
        "arrayfire::cuda::resize", {{resize_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(method)));

    dim3 threads(TX, TY, 1);
    dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));
    int blocksPerMatX = blocks.x;
    int blocksPerMatY = blocks.y;

    if (in.dims[2] > 1) { blocks.x *= in.dims[2]; }
    if (in.dims[3] > 1) { blocks.y *= in.dims[3]; }
    float xf = (float)in.dims[0] / (float)out.dims[0];
    float yf = (float)in.dims[1] / (float)out.dims[1];

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    resize(qArgs, out, in, blocksPerMatX, blocksPerMatY, xf, yf);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
