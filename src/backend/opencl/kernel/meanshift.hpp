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
#include <kernel_headers/meanshift.hpp>
#include <traits.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void meanshift(Param out, const Param in, const float spatialSigma,
               const float chromaticSigma, const uint numIters,
               const bool is_color) {
    using AccType = typename std::conditional<std::is_same<T, double>::value,
                                              double, float>::type;
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(is_color),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(AccType, dtype_traits<AccType>::getName()),
        DefineKeyValue(MAX_CHANNELS, (is_color ? 3 : 1)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto meanshiftOp =
        common::getKernel("meanshift", {{meanshift_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    const int bCount = (is_color ? 1 : in.info.dims[2]);

    cl::NDRange global(bCount * blk_x * THREADS_X,
                       in.info.dims[3] * blk_y * THREADS_Y);

    // clamp spatical and chromatic sigma's
    int radius = std::max((int)(spatialSigma * 1.5f), 1);

    const float cvar = chromaticSigma * chromaticSigma;

    meanshiftOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *in.data, in.info, radius, cvar, numIters, blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
