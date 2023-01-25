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
#include <kernel_headers/triangle.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void triangle(Param out, const Param in, bool is_upper, bool is_unit_diag) {
    using arrayfire::opencl::scalar_to_option;
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 128;
    constexpr unsigned TILEY = 32;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(is_upper),
        TemplateArg(is_unit_diag),
    };
    vector<string> compileOpts = {
        DefineValue(is_upper),
        DefineValue(is_unit_diag),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        DefineKeyValue(ONE, scalar_to_option(scalar<T>(1))),
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto triangle = common::getKernel("triangle", {{triangle_cl_src}},
                                      tmpltArgs, compileOpts);

    NDRange local(TX, TY);

    int groups_x = divup(out.info.dims[0], TILEX);
    int groups_y = divup(out.info.dims[1], TILEY);

    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    triangle(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, groups_x, groups_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
