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
#include <kernel_headers/histogram.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

template<typename inType, typename outType>
void histogram(Param out, const Param in, int nbins, float minval, float maxval,
               bool isLinear) {
    constexpr int MAX_BINS  = 4000;
    constexpr int THREADS_X = 256;
    constexpr int THRD_LOAD = 16;

    static const std::string src(histogram_cl, histogram_cl_len);

    std::vector<TemplateArg> targs = {
        TemplateTypename<inType>(),
        TemplateTypename<outType>(),
        TemplateArg(isLinear),
    };
    std::vector<std::string> options = {
        DefineKeyValue(inType, dtype_traits<inType>::getName()),
        DefineKeyValue(outType, dtype_traits<outType>::getName()),
        DefineValue(THRD_LOAD),
        DefineValue(MAX_BINS),
    };
    options.emplace_back(getTypeBuildDefinition<inType>());
    if (isLinear) { options.emplace_back(DefineKey(IS_LINEAR)); }

    auto histogram = common::findKernel("histogram", {src}, targs, options);

    int nElems  = in.info.dims[0] * in.info.dims[1];
    int blk_x   = divup(nElems, THRD_LOAD * THREADS_X);
    int locSize = nbins <= MAX_BINS ? (nbins * sizeof(outType)) : 1;

    cl::NDRange local(THREADS_X, 1);
    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X, in.info.dims[3]);

    histogram(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, cl::Local(locSize), nElems, nbins, minval,
              maxval, blk_x);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
