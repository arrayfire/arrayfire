/*******************************************************
 * Copyright (c) 2017, ArrayFire
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
#include <kernel_headers/anisotropic_diffusion.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, bool isMCDE>
void anisotropicDiffusion(Param inout, const float dt, const float mct,
                          const int fluxFnCode) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;
    constexpr int YDIM_LOAD = 2 * THREADS_X / THREADS_Y;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(isMCDE),
        TemplateArg(fluxFnCode),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(SHRD_MEM_HEIGHT, (THREADS_Y * YDIM_LOAD + 2)),
        DefineKeyValue(SHRD_MEM_WIDTH, (THREADS_X + 2)),
        DefineKeyValue(IS_MCDE, isMCDE),
        DefineKeyValue(FLUX_FN, fluxFnCode),
        DefineValue(YDIM_LOAD),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto diffUpdate =
        common::getKernel("aisoDiffUpdate", {{anisotropic_diffusion_cl_src}},
                          tmpltArgs, compileOpts);

    NDRange local(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.info.dims[0], local[0]);
    int blkY = divup(inout.info.dims[1], local[1] * YDIM_LOAD);

    NDRange global(local[0] * blkX * inout.info.dims[2],
                   local[1] * blkY * inout.info.dims[3], 1);

    diffUpdate(EnqueueArgs(getQueue(), global, local), *inout.data, inout.info,
               dt, mct, blkX, blkY);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
