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
#include <kernel_headers/unwrap.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void unwrap(Param out, const Param in, const dim_t wx, const dim_t wy,
            const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
            const dim_t dx, const dim_t dy, const dim_t nx,
            const bool is_column) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    ToNumStr<T> toNumStr;
    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(is_column),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(IS_COLUMN, is_column),
        DefineKeyValue(ZERO, toNumStr(scalar<T>(0))),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto unwrap =
        common::getKernel("unwrap", {{unwrap_cl_src}}, tmpltArgs, compileOpts);

    dim_t TX = 1, TY = 1;
    dim_t BX       = 1;
    const dim_t BY = out.info.dims[2] * out.info.dims[3];
    int reps       = 1;

    if (is_column) {
        TX   = std::min(THREADS_PER_GROUP, nextpow2(out.info.dims[0]));
        TY   = THREADS_PER_GROUP / TX;
        BX   = divup(out.info.dims[1], TY);
        reps = divup((wx * wy), TX);
    } else {
        TX   = THREADS_X;
        TY   = THREADS_Y;
        BX   = divup(out.info.dims[0], TX);
        reps = divup((wx * wy), TY);
    }

    NDRange local(TX, TY);
    NDRange global(local[0] * BX, local[1] * BY);

    unwrap(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *in.data, in.info, static_cast<int>(wx), static_cast<int>(wy),
           static_cast<int>(sx), static_cast<int>(sy), static_cast<int>(px),
           static_cast<int>(py), static_cast<int>(dx), static_cast<int>(dy),
           static_cast<int>(nx), reps);
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
