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
#include <nvrtc_kernel_headers/histogram.hpp>

#include <string>

namespace cuda {
namespace kernel {

constexpr int MAX_BINS  = 4000;
constexpr int THREADS_X = 256;
constexpr int THRD_LOAD = 16;

template<typename inType, typename outType>
void histogram(Param<outType> out, CParam<inType> in, int nbins, float minval,
               float maxval, bool isLinear) {
    static const std::string source(histogram_cuh, histogram_cuh_len);
    // clang-format off
    auto histogram = getKernel("cuda::histogram", source,
            {
              TemplateTypename<inType>(),
              TemplateTypename<outType>(),
              TemplateArg(isLinear)
            },
            {
              DefineValue(MAX_BINS),
              DefineValue(THRD_LOAD)
            }
            );
    // clang-format on
    dim3 threads(kernel::THREADS_X, 1);

    int nElems = in.dims[0] * in.dims[1];
    int blk_x  = divup(nElems, THRD_LOAD * THREADS_X);

    dim3 blocks(blk_x * in.dims[2], in.dims[3]);

    // If nbins > MAX_BINS, we are using global memory so smem_size can be 0;
    int smem_size = nbins <= MAX_BINS ? (nbins * sizeof(outType)) : 0;

    EnqueueArgs qArgs(blocks, threads, getActiveStream(), smem_size);
    histogram(qArgs, out, in, nElems, nbins, minval, maxval, blk_x);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
