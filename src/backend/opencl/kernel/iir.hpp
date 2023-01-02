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
#include <kernel_headers/iir.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, bool batch_a>
void iir(Param y, Param c, Param a) {
    // FIXME: This is a temporary fix. Ideally the local memory should be
    // allocted outside
    constexpr int MAX_A_SIZE = (1024 * sizeof(double)) / sizeof(T);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(batch_a),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(MAX_A_SIZE),
        DefineKeyValue(BATCH_A, batch_a),
        DefineKeyValue(ZERO, scalar_to_option(scalar<T>(0))),
        getTypeBuildDefinition<T>()};

    auto iir = common::getKernel("iir_kernel", {{iir_cl_src}}, targs, options);

    const int groups_y = y.info.dims[1];
    const int groups_x = y.info.dims[2];

    int threads = 256;
    while (threads > (int)y.info.dims[0] && threads > 32) threads /= 2;

    cl::NDRange local(threads, 1);
    cl::NDRange global(groups_x * local[0],
                       groups_y * y.info.dims[3] * local[1]);

    try {
        iir(cl::EnqueueArgs(getQueue(), global, local), *y.data, y.info,
            *c.data, c.info, *a.data, a.info, groups_y);
    } catch (cl::Error& clerr) {
        AF_ERROR("Size of a too big for this datatype", AF_ERR_SIZE);
    }
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
