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
#include <nvrtc_kernel_headers/meanshift_cuh.hpp>

#include <type_traits>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T>
void meanshift(Param<T> out, CParam<T> in, const float spatialSigma,
               const float chromaticSigma, const uint numIters, bool IsColor) {
    typedef typename std::conditional<std::is_same<T, double>::value, double,
                                      float>::type AccType;
    auto meanshift = common::getKernel(
        "arrayfire::cuda::meanshift", {{meanshift_cuh_src}},
        TemplateArgs(TemplateTypename<AccType>(), TemplateTypename<T>(),
                     TemplateArg((IsColor ? 3 : 1))  // channels
                     ));

    static dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x        = divup(in.dims[0], THREADS_X);
    int blk_y        = divup(in.dims[1], THREADS_Y);
    const int bCount = (IsColor ? 1 : in.dims[2]);

    dim3 blocks(blk_x * bCount, blk_y * in.dims[3]);

    // clamp spatical and chromatic sigma's
    int radius       = std::max((int)(spatialSigma * 1.5f), 1);
    const float cvar = chromaticSigma * chromaticSigma;

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    meanshift(qArgs, out, in, radius, cvar, numIters, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
