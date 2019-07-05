/*******************************************************
 * Copyright (c) 2016, ArrayFire
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
#include <nvrtc_kernel_headers/moments_cuh.hpp>
#include <af/defines.h>

#include <string>

namespace cuda {
namespace kernel {

static const int THREADS = 128;

template<typename T>
void moments(Param<float> out, CParam<T> in, const af::momentType moment) {
    static const std::string source(moments_cuh, moments_cuh_len);

    auto moments = getKernel("cuda::moments", source, {TemplateTypename<T>()});

    dim3 threads(THREADS, 1, 1);
    dim3 blocks(in.dims[1], in.dims[2] * in.dims[3]);

    bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream(),
                      sizeof(float) * out.dims[0]);

    moments(qArgs, out, in, moment, pBatch);

    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
