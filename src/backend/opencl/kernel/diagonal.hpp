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
#include <kernel_headers/diag_create.hpp>
#include <kernel_headers/diag_extract.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
static void diagCreate(Param out, Param in, int num) {
    std::array<TemplateArg, 1> targs = {
        TemplateTypename<T>(),
    };
    std::array<std::string, 3> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto diagCreate = common::getKernel("diagCreateKernel",
                                        {{diag_create_cl_src}}, targs, options);

    cl::NDRange local(32, 8);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    cl::NDRange global(groups_x * local[0] * out.info.dims[2],
                       groups_y * local[1]);

    diagCreate(cl::EnqueueArgs(getQueue(), global, local), *(out.data),
               out.info, *(in.data), in.info, num, groups_x);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
static void diagExtract(Param out, Param in, int num) {
    std::array<TemplateArg, 1> targs = {
        TemplateTypename<T>(),
    };
    std::array<std::string, 3> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto diagExtract = common::getKernel(
        "diagExtractKernel", {{diag_extract_cl_src}}, targs, options);

    cl::NDRange local(256, 1);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_z = out.info.dims[2];
    cl::NDRange global(groups_x * local[0],
                       groups_z * local[1] * out.info.dims[3]);

    diagExtract(cl::EnqueueArgs(getQueue(), global, local), *(out.data),
                out.info, *(in.data), in.info, num, groups_z);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
