/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
#include <kernel_headers/pad_array_borders.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
static const int PADB_THREADS_X = 16;
static const int PADB_THREADS_Y = 16;

template<typename T>
void padBorders(Param out, const Param in, dim4 const& lBPadding,
                const af_border_type borderType) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(borderType),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(AF_BORDER_TYPE, (int)borderType),
        DefineKeyValue(AF_PAD_SYM, (int)AF_PAD_SYM),
        DefineKeyValue(AF_PAD_PERIODIC, (int)AF_PAD_PERIODIC),
        DefineKeyValue(AF_PAD_CLAMP_TO_EDGE, (int)AF_PAD_CLAMP_TO_EDGE),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto pad = common::getKernel("padBorders", {{pad_array_borders_cl_src}},
                                 tmpltArgs, compileOpts);

    NDRange local(PADB_THREADS_X, PADB_THREADS_Y);

    unsigned blk_x = divup(out.info.dims[0], local[0]);
    unsigned blk_y = divup(out.info.dims[1], local[1]);

    NDRange global(blk_x * out.info.dims[2] * local[0],
                   blk_y * out.info.dims[3] * local[1]);

    pad(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
        in.info, static_cast<int>(lBPadding[0]), static_cast<int>(lBPadding[1]),
        static_cast<int>(lBPadding[2]), static_cast<int>(lBPadding[3]), blk_x,
        blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
