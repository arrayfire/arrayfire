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
#include <kernel_headers/tile.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void tile(Param out, const Param in) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr int TX    = 32;
    constexpr int TY    = 8;
    constexpr int TILEX = 512;
    constexpr int TILEY = 32;

    vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto tile = common::getKernel("tile", {{tile_cl_src}}, targs, compileOpts);

    NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    tile(EnqueueArgs(getQueue(), global, local), *out.data, *in.data, out.info,
         in.info, blocksPerMatX, blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
