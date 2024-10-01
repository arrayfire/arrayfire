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
#include <nvrtc_kernel_headers/sparse_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

template<typename T>
void coo2dense(Param<T> output, CParam<T> values, CParam<int> rowIdx,
               CParam<int> colIdx) {
    constexpr int reps = 4;

    auto coo2Dense = common::getKernel(
        "arrayfire::cuda::coo2Dense", {{sparse_cuh_src}},
        TemplateArgs(TemplateTypename<T>()), {{DefineValue(reps)}});

    dim3 threads(256, 1, 1);

    dim3 blocks(divup(values.dims[0], threads.x * reps), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    coo2Dense(qArgs, output, values, rowIdx, colIdx);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
