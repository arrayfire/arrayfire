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
#include <kernel_headers/iota.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void iota(Param out, const af::dim4& sdims) {
    constexpr int IOTA_TX = 32;
    constexpr int IOTA_TY = 8;
    constexpr int TILEX   = 512;
    constexpr int TILEY   = 32;

    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto iota = common::getKernel("iota_kernel", {{iota_cl_src}},
                                  TemplateArgs(TemplateTypename<T>()), options);
    cl::NDRange local(IOTA_TX, IOTA_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    cl::NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                       local[1] * blocksPerMatY * out.info.dims[3], 1);

    iota(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
         static_cast<int>(sdims[0]), static_cast<int>(sdims[1]),
         static_cast<int>(sdims[2]), static_cast<int>(sdims[3]), blocksPerMatX,
         blocksPerMatY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
