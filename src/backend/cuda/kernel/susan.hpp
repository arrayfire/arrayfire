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
#include <nvrtc_kernel_headers/susan_cuh.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

constexpr unsigned BLOCK_X = 16;
constexpr unsigned BLOCK_Y = 16;

template<typename T>
void susan_responses(T* out, const T* in, const unsigned idim0,
                     const unsigned idim1, const int radius, const float t,
                     const float g, const unsigned edge) {
    auto susan =
        common::getKernel("arrayfire::cuda::susan", {{susan_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()),
                          {{DefineValue(BLOCK_X), DefineValue(BLOCK_Y)}});

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(divup(idim0 - edge * 2, BLOCK_X),
                divup(idim1 - edge * 2, BLOCK_Y));
    const size_t SMEM_SIZE =
        (BLOCK_X + 2 * radius) * (BLOCK_Y + 2 * radius) * sizeof(T);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), SMEM_SIZE);

    susan(qArgs, out, in, idim0, idim1, radius, t, g, edge);
    POST_LAUNCH_CHECK();
}

template<typename T>
void nonMaximal(float* x_out, float* y_out, float* resp_out, unsigned* count,
                const unsigned idim0, const unsigned idim1, const T* resp_in,
                const unsigned edge, const unsigned max_corners) {
    auto nonMax =
        common::getKernel("arrayfire::cuda::nonMax", {{susan_cuh_src}},
                          TemplateArgs(TemplateTypename<T>()));

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(divup(idim0 - edge * 2, BLOCK_X),
                divup(idim1 - edge * 2, BLOCK_Y));

    auto d_corners_found = memAlloc<unsigned>(1);
    CUDA_CHECK(cudaMemsetAsync(d_corners_found.get(), 0, sizeof(unsigned),
                               getActiveStream()));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    nonMax(qArgs, x_out, y_out, resp_out, d_corners_found.get(), idim0, idim1,
           resp_in, edge, max_corners);
    POST_LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(count, d_corners_found.get(), sizeof(unsigned),
                               cudaMemcpyDeviceToHost, getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
