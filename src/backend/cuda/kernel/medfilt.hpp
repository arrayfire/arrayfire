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
#include <nvrtc_kernel_headers/medfilt_cuh.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int MAX_MEDFILTER1_LEN = 121;
static const int MAX_MEDFILTER2_LEN = 15;
static const int THREADS_X          = 16;
static const int THREADS_Y          = 16;

template<typename T>
void medfilt2(Param<T> out, CParam<T> in, const af::borderType pad, int w_len,
              int w_wid) {
    UNUSED(w_wid);
    auto medfilt2 =
        common::getKernel("arrayfire::cuda::medfilt2", {{medfilt_cuh_src}},
                          TemplateArgs(TemplateTypename<T>(), TemplateArg(pad),
                                       TemplateArg(w_len), TemplateArg(w_wid)),
                          {{DefineValue(THREADS_X), DefineValue(THREADS_Y)}});

    const dim3 threads(THREADS_X, THREADS_Y);

    int blk_x = divup(in.dims[0], threads.x);
    int blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    medfilt2(qArgs, out, in, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

template<typename T>
void medfilt1(Param<T> out, CParam<T> in, const af::borderType pad, int w_wid) {
    auto medfilt1 =
        common::getKernel("arrayfire::cuda::medfilt1", {{medfilt_cuh_src}},
                          TemplateArgs(TemplateTypename<T>(), TemplateArg(pad),
                                       TemplateArg(w_wid)));

    const dim3 threads(THREADS_X);

    int blk_x = divup(in.dims[0], threads.x);

    dim3 blocks(blk_x * in.dims[1], in.dims[2], in.dims[3]);

    const size_t shrdMemBytes = sizeof(T) * (THREADS_X + w_wid - 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), shrdMemBytes);
    medfilt1(qArgs, out, in, w_wid, blk_x);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
