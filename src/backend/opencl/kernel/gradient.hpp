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
#include <kernel/config.hpp>
#include <kernel_headers/gradient.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void gradient(Param grad0, Param grad1, const Param in) {
    constexpr int TX = 32;
    constexpr int TY = 8;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options{
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(TX),
        DefineValue(TY),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        DefineKeyValue(CPLX, static_cast<int>(iscplx<T>())),
        getTypeBuildDefinition<T>()};

    auto gradOp =
        common::getKernel("gradient", {{gradient_cl_src}}, targs, options);

    cl::NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(in.info.dims[0], TX);
    int blocksPerMatY = divup(in.info.dims[1], TY);
    cl::NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                       local[1] * blocksPerMatY * in.info.dims[3], 1);

    gradOp(cl::EnqueueArgs(getQueue(), global, local), *grad0.data, grad0.info,
           *grad1.data, grad1.info, *in.data, in.info, blocksPerMatX,
           blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
