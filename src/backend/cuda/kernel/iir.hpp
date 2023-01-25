/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/iir_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T, bool batch_a>
void iir(Param<T> y, CParam<T> c, CParam<T> a) {
    constexpr int MAX_A_SIZE = 1024;

    auto iir = common::getKernel(
        "arrayfire::cuda::iir", {{iir_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(batch_a)),
        {{DefineValue(MAX_A_SIZE)}});

    const int blocks_y = y.dims[1];
    const int blocks_x = y.dims[2];

    dim3 blocks(blocks_x, blocks_y * y.dims[3]);

    int threads = 256;
    while (threads > y.dims[0] && threads > 32) threads /= 2;

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    iir(qArgs, y, c, a, blocks_y);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
