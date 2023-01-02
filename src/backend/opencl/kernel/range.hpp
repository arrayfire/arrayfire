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
#include <kernel_headers/range.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void range(Param out, const int dim) {
    constexpr int RANGE_TX    = 32;
    constexpr int RANGE_TY    = 8;
    constexpr int RANGE_TILEX = 512;
    constexpr int RANGE_TILEY = 32;

    std::vector<TemplateArg> targs   = {TemplateTypename<T>()};
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto rangeOp =
        common::getKernel("range_kernel", {{range_cl_src}}, targs, options);

    cl::NDRange local(RANGE_TX, RANGE_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.info.dims[1], RANGE_TILEY);
    cl::NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                       local[1] * blocksPerMatY * out.info.dims[3], 1);

    rangeOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            dim, blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
