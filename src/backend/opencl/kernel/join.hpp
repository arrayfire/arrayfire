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
#include <kernel_headers/join.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

template<typename T>
void join(Param out, const Param in, dim_t dim, const af::dim4 offset) {
    constexpr int TX    = 32;
    constexpr int TY    = 8;
    constexpr int TILEX = 256;
    constexpr int TILEY = 32;

    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto join =
        common::getKernel("join_kernel", {join_cl_src},
                          {TemplateTypename<T>(), TemplateArg(dim)}, options);
    cl::NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(in.info.dims[0], TILEX);
    int blocksPerMatY = divup(in.info.dims[1], TILEY);
    cl::NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                       local[1] * blocksPerMatY * in.info.dims[3], 1);

    join(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
         *in.data, in.info, static_cast<int>(offset[0]),
         static_cast<int>(offset[1]), static_cast<int>(offset[2]),
         static_cast<int>(offset[3]), blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
