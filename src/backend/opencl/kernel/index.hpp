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

namespace opencl {
namespace kernel {

typedef struct {
    int offs[4];
    int strds[4];
    char isSeq[4];
} IndexKernelParam_t;

template<typename T>
void index(Param out, const Param in, const IndexKernelParam_t& p,
           cl::Buffer* bPtr[4]) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    static const std::string src(index_cl, index_cl_len);

    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto index = common::findKernel("indexKernel", {src},
                                    {TemplateTypename<T>()}, options);
    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * out.info.dims[2] * THREADS_X,
                       blk_y * out.info.dims[3] * THREADS_Y);

    index(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *in.data, in.info, p, *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3], blk_x,
          blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
