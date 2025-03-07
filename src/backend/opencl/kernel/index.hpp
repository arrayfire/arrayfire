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
#include <debug_opencl.hpp>
#include <kernel_headers/index.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

typedef struct {
    int offs[4];
    int strds[4];
    int steps[4];
    char isSeq[4];
} IndexKernelParam_t;

template<typename T>
void index(Param out, const Param in, const IndexKernelParam_t& p,
           cl::Buffer* bPtr[4]) {
    std::array<std::string, 2> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        getTypeBuildDefinition<T>()};

    auto index =
        common::getKernel("indexKernel", {{index_cl_src}},
                          TemplateArgs(TemplateTypename<T>()), options);
    int threads_x = 256;
    int threads_y = 1;
    cl::NDRange local(threads_x, threads_y);
    switch (out.info.dims[1]) {
        case 1: threads_y = 1; break;
        case 2: threads_y = 2; break;
        case 3:
        case 4: threads_y = 4; break;
        default: threads_y = 8; break;
    }
    threads_x = static_cast<unsigned>(256.f / threads_y);

    int blk_x = divup(out.info.dims[0], local[0]);
    int blk_y = divup(out.info.dims[1], local[1]);

    cl::NDRange global(blk_x * out.info.dims[2] * local[0],
                       blk_y * out.info.dims[3] * local[1]);

    index(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *in.data, in.info, p, *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3], blk_x,
          blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
