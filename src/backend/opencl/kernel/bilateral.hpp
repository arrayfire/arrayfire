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
#include <kernel_headers/bilateral.hpp>
#include <traits.hpp>
#include <af/opencl.h>

#include <algorithm>
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename inType, typename outType>
void bilateral(Param out, const Param in, const float s_sigma,
               const float c_sigma) {
    constexpr int THREADS_X     = 16;
    constexpr int THREADS_Y     = 16;
    constexpr bool UseNativeExp = !std::is_same<inType, double>::value ||
                                  std::is_same<inType, cdouble>::value;

    std::vector<TemplateArg> targs = {
        TemplateTypename<inType>(),
        TemplateTypename<outType>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(inType, dtype_traits<inType>::getName()),
        DefineKeyValue(outType, dtype_traits<outType>::getName()),
    };
    if (UseNativeExp) { options.emplace_back(DefineKey(USE_NATIVE_EXP)); }
    options.emplace_back(getTypeBuildDefinition<inType>());

    auto bilateralOp =
        common::getKernel("bilateral", {{bilateral_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);

    // calculate local memory size
    int radius          = (int)std::max(s_sigma * 1.5f, 1.f);
    int num_shrd_elems  = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    int num_gauss_elems = (2 * radius + 1) * (2 * radius + 1);
    size_t localMemSize = (num_shrd_elems + num_gauss_elems) * sizeof(outType);
    size_t MaxLocalSize =
        getDevice(getActiveDeviceId()).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    if (localMemSize > MaxLocalSize) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOpenCL Bilateral filter doesn't support %f spatial sigma\n",
                 s_sigma);
        OPENCL_NOT_SUPPORTED(errMessage);
    }

    bilateralOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *in.data, in.info, cl::Local(num_shrd_elems * sizeof(outType)),
                cl::Local(num_gauss_elems * sizeof(outType)), s_sigma, c_sigma,
                num_shrd_elems, blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
