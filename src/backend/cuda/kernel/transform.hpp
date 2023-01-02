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
#include <nvrtc_kernel_headers/transform_cuh.hpp>
#include <af/defines.h>

#include <algorithm>

namespace arrayfire {
namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const unsigned TX = 16;
static const unsigned TY = 16;
// Used for batching images
static const unsigned TI = 4;

template<typename T>
void transform(Param<T> out, CParam<T> in, CParam<float> tf, const bool inverse,
               const bool perspective, const af::interpType method, int order) {
    auto transform = common::getKernel(
        "arrayfire::cuda::transform", {{transform_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(inverse),
                     TemplateArg(order)));

    const unsigned int nImg2  = in.dims[2];
    const unsigned int nImg3  = in.dims[3];
    const unsigned int nTfs2  = tf.dims[2];
    const unsigned int nTfs3  = tf.dims[3];
    const unsigned int tf_len = (perspective) ? 9 : 6;

    // Copy transform to constant memory.
    auto constPtr = transform.getDevPtr("c_tmat");
    transform.copyToReadOnly(constPtr, reinterpret_cast<CUdeviceptr>(tf.ptr),
                             nTfs2 * nTfs3 * tf_len * sizeof(float));

    dim3 threads(TX, TY, 1);
    dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

    const int blocksXPerImage = blocks.x;
    const int blocksYPerImage = blocks.y;

    // Takes care of all types of batching
    // One-to-one batching is only done on blocks.x
    // TODO If dim2 is not one-to-one batched, then divide blocks.x by factor
    int batchImg2 = 1;
    if (nImg2 != nTfs2) batchImg2 = std::min(nImg2, TI);

    blocks.x *= (nImg2 / batchImg2);
    blocks.y *= nImg3;

    // Use blocks.z for transforms
    blocks.z *= std::max((nTfs2 / nImg2), 1u) * std::max((nTfs3 / nImg3), 1u);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    transform(qArgs, out, in, nImg2, nImg3, nTfs2, nTfs3, batchImg2,
              blocksXPerImage, blocksYPerImage, perspective, method);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
