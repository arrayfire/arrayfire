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
#include <common/half.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/lookup.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename in_t, typename idx_t>
void lookup(Param out, const Param in, const Param indices,
            const unsigned dim) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    std::vector<TemplateArg> targs = {
        TemplateTypename<in_t>(),
        TemplateTypename<idx_t>(),
        TemplateArg(dim),
    };
    std::vector<std::string> options = {
        DefineKeyValue(in_t, dtype_traits<in_t>::getName()),
        DefineKeyValue(idx_t, dtype_traits<idx_t>::getName()),
        DefineKeyValue(DIM, dim),
    };
    options.emplace_back(getTypeBuildDefinition<in_t, idx_t>());

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * out.info.dims[2] * THREADS_X,
                       blk_y * out.info.dims[3] * THREADS_Y);

    auto arrIdxOp =
        common::getKernel("lookupND", {{lookup_cl_src}}, targs, options);

    arrIdxOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, *indices.data, indices.info, blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
