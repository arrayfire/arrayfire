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
#include <kernel_headers/sobel.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename Ti, typename To, unsigned ker_size>
void sobel(Param dx, Param dy, const Param in) {
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    std::vector<TemplateArg> targs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(ker_size),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(KER_SIZE, ker_size),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    auto sobel =
        common::getKernel("sobel3x3", {{sobel_cl_src}}, targs, compileOpts);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);
    size_t loc_size =
        (THREADS_X + ker_size - 1) * (THREADS_Y + ker_size - 1) * sizeof(Ti);

    sobel(cl::EnqueueArgs(getQueue(), global, local), *dx.data, dx.info,
          *dy.data, dy.info, *in.data, in.info, cl::Local(loc_size), blk_x,
          blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
