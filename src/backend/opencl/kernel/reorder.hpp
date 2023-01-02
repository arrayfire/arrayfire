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
#include <kernel_headers/reorder.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void reorder(Param out, const Param in, const dim_t* rdims) {
    constexpr int TX    = 32;
    constexpr int TY    = 8;
    constexpr int TILEX = 512;
    constexpr int TILEY = 32;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto reorderOp =
        common::getKernel("reorder_kernel", {{reorder_cl_src}}, targs, options);

    cl::NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    cl::NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                       local[1] * blocksPerMatY * out.info.dims[3], 1);

    reorderOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, *in.data,
              out.info, in.info, static_cast<int>(rdims[0]),
              static_cast<int>(rdims[1]), static_cast<int>(rdims[2]),
              static_cast<int>(rdims[3]), blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
