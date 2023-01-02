/*******************************************************
 * Copyright (c) 2016, ArrayFire
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
#include <kernel_headers/moments.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void moments(Param out, const Param in, af_moment_type moment) {
    constexpr int THREADS = 128;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(out.info.dims[0]),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(MOMENTS_SZ, out.info.dims[0]),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto momentsOp =
        common::getKernel("moments", {{moments_cl_src}}, targs, options);

    cl::NDRange local(THREADS, 1, 1);
    cl::NDRange global(in.info.dims[1] * local[0],
                       in.info.dims[2] * in.info.dims[3] * local[1]);

    bool pBatch = !(in.info.dims[2] == 1 && in.info.dims[3] == 1);

    momentsOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, (int)moment, (int)pBatch);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
