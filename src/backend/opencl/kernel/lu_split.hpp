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
#include <kernel_headers/lu_split.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void luSplitLauncher(Param lower, Param upper, const Param in, bool same_dims) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 128;
    constexpr unsigned TILEY = 32;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(same_dims),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(same_dims),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        DefineKeyValue(ONE, scalar_to_option(scalar<T>(1))),
        getTypeBuildDefinition<T>()};

    auto luSplit =
        common::getKernel("luSplit", {{lu_split_cl_src}}, targs, options);

    cl::NDRange local(TX, TY);

    int groups_x = divup(in.info.dims[0], TILEX);
    int groups_y = divup(in.info.dims[1], TILEY);

    cl::NDRange global(groups_x * local[0] * in.info.dims[2],
                       groups_y * local[1] * in.info.dims[3]);

    luSplit(cl::EnqueueArgs(getQueue(), global, local), *lower.data, lower.info,
            *upper.data, upper.info, *in.data, in.info, groups_x, groups_y);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void luSplit(Param lower, Param upper, const Param in) {
    bool same_dims = (lower.info.dims[0] == in.info.dims[0]) &&
                     (lower.info.dims[1] == in.info.dims[1]);
    luSplitLauncher<T>(lower, upper, in, same_dims);
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
