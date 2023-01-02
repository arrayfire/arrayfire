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
#include <nvrtc_kernel_headers/bilateral_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename inType, typename outType>
void bilateral(Param<outType> out, CParam<inType> in, float s_sigma,
               float c_sigma) {
    auto bilateral = common::getKernel(
        "arrayfire::cuda::bilateral", {{bilateral_cuh_src}},
        TemplateArgs(TemplateTypename<inType>(), TemplateTypename<outType>()),
        {{DefineValue(THREADS_X), DefineValue(THREADS_Y)}});

    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    int blk_x = divup(in.dims[0], THREADS_X);
    int blk_y = divup(in.dims[1], THREADS_Y);

    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    // calculate shared memory size
    int radius          = (int)std::max(s_sigma * 1.5f, 1.f);
    int num_shrd_elems  = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    int num_gauss_elems = (2 * radius + 1) * (2 * radius + 1);
    size_t total_shrd_size =
        sizeof(outType) * (num_shrd_elems + num_gauss_elems);

    size_t MAX_SHRD_SIZE = getDeviceProp(getActiveDeviceId()).sharedMemPerBlock;
    if (total_shrd_size > MAX_SHRD_SIZE) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nCUDA Bilateral filter doesn't support %f spatial sigma\n",
                 s_sigma);
        CUDA_NOT_SUPPORTED(errMessage);
    }

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), total_shrd_size);

    bilateral(qArgs, out, in, s_sigma, c_sigma, num_shrd_elems, blk_x, blk_y);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
