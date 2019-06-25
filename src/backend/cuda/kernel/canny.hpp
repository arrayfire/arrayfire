/*******************************************************
 * Copyright (c) 2017, ArrayFire
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
#include <nvrtc_kernel_headers/canny.hpp>

#include <string>

namespace cuda {
namespace kernel {

static const int STRONG = 1;
static const int WEAK   = 2;
static const int NOEDGE = 0;

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T>
void nonMaxSuppression(Param<T> output, CParam<T> magnitude, CParam<T> dx,
                       CParam<T> dy) {
    static const std::string source(canny_cuh, canny_cuh_len);

    // clang-format off
    auto maxSuppress = getKernel("cuda::nonMaxSuppression", source,
            {
              TemplateTypename<T>()
            },
            {
              DefineValue(STRONG),
              DefineValue(WEAK),
              DefineValue(NOEDGE),
              DefineValue(THREADS_X),
              DefineValue(THREADS_Y)
            }
            );
    // clang-format on

    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    // Launch only threads to process non-border pixels
    int blk_x = divup(magnitude.dims[0] - 2, threads.x);
    int blk_y = divup(magnitude.dims[1] - 2, threads.y);

    // launch batch * blk_x blocks along x dimension
    dim3 blocks(blk_x * magnitude.dims[2], blk_y * magnitude.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    maxSuppress(qArgs, output, magnitude, dx, dy, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}

template<typename T>
void edgeTrackingHysteresis(Param<T> output, CParam<T> strong, CParam<T> weak) {
    static const std::string source(canny_cuh, canny_cuh_len);

    // clang-format off
    auto init = getKernel("cuda::initEdgeOut", source,
            {
              TemplateTypename<T>()
            },
            {
              DefineValue(STRONG),
              DefineValue(WEAK),
              DefineValue(NOEDGE),
              DefineValue(THREADS_X),
              DefineValue(THREADS_Y)
            }
            );
    auto trace = getKernel("cuda::edgeTrack", source,
            {
              TemplateTypename<T>()
            },
            {
              DefineValue(STRONG),
              DefineValue(WEAK),
              DefineValue(NOEDGE),
              DefineValue(THREADS_X),
              DefineValue(THREADS_Y)
            }
            );
    auto threshold = getKernel("cuda::suppressLeftOver", source,
            {
              TemplateTypename<T>()
            },
            {
              DefineValue(STRONG),
              DefineValue(WEAK),
              DefineValue(NOEDGE),
              DefineValue(THREADS_X),
              DefineValue(THREADS_Y)
            }
            );
    // clang-format on

    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    // Launch only threads to process non-border pixels
    int blk_x = divup(weak.dims[0] - 2, threads.x);
    int blk_y = divup(weak.dims[1] - 2, threads.y);

    // launch batch * blk_x blocks along x dimension
    dim3 blocks(blk_x * weak.dims[2], blk_y * weak.dims[3]);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    init(qArgs, output, strong, weak, blk_x, blk_y);
    POST_LAUNCH_CHECK();

    int notFinished = 1;
    while (notFinished) {
        notFinished = 0;
        trace.setScalar("hasChanged", notFinished);
        trace(qArgs, output, blk_x, blk_y);
        POST_LAUNCH_CHECK();
        trace.getScalar(notFinished, "hasChanged");
    }
    threshold(qArgs, output, blk_x, blk_y);
    POST_LAUNCH_CHECK();
}
}  // namespace kernel
}  // namespace cuda
