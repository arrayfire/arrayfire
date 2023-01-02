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
#include <kernel/config.hpp>
#include <kernel_headers/identity.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
static void identity(Param out) {
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ONE, scalar_to_option(scalar<T>(1))),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto identityOp = common::getKernel("identity_kernel", {{identity_cl_src}},
                                        targs, options);

    cl::NDRange local(32, 8);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    cl::NDRange global(groups_x * out.info.dims[2] * local[0],
                       groups_y * out.info.dims[3] * local[1]);

    identityOp(cl::EnqueueArgs(getQueue(), global, local), *(out.data),
               out.info, groups_x, groups_y);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
