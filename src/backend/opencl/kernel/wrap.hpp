/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <kernel_headers/wrap.hpp>
#include <kernel_headers/wrap_dilated.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void wrap(Param out, const Param in, const dim_t wx, const dim_t wy,
          const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
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
        DefineValue(is_column),
        DefineKeyValue(ZERO, toNumStr(scalar<T>(0))),
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto wrap =
        common::getKernel("wrap", {{wrap_cl_src}}, tmpltArgs, compileOpts);

    dim_t nx = (out.info.dims[0] + 2 * px - wx) / sx + 1;
    dim_t ny = (out.info.dims[1] + 2 * py - wy) / sy + 1;

    NDRange local(THREADS_X, THREADS_Y);

    dim_t groups_x = divup(out.info.dims[0], local[0]);
    dim_t groups_y = divup(out.info.dims[1], local[1]);

    NDRange global(local[0] * groups_x * out.info.dims[2],
                   local[1] * groups_y * out.info.dims[3]);

    wrap(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
         in.info, static_cast<int>(wx), static_cast<int>(wy),
         static_cast<int>(sx), static_cast<int>(sy), static_cast<int>(px),
         static_cast<int>(py), static_cast<int>(nx), static_cast<int>(ny),
         static_cast<int>(groups_x), static_cast<int>(groups_y));

    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void wrap_dilated(Param out, const Param in, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy, const dim_t px,
                  const dim_t py, const dim_t dx, const dim_t dy,
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
        DefineValue(is_column),
        DefineKeyValue(ZERO, toNumStr(scalar<T>(0))),
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto dilatedWrap = common::getKernel(
        "wrap_dilated", {{wrap_dilated_cl_src}}, tmpltArgs, compileOpts);

    dim_t nx = 1 + (out.info.dims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    dim_t ny = 1 + (out.info.dims[1] + 2 * py - (((wy - 1) * dy) + 1)) / sy;

    NDRange local(THREADS_X, THREADS_Y);

    dim_t groups_x = divup(out.info.dims[0], local[0]);
    dim_t groups_y = divup(out.info.dims[1], local[1]);

    NDRange global(local[0] * groups_x * out.info.dims[2],
                   local[1] * groups_y * out.info.dims[3]);

    dilatedWrap(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *in.data, in.info, static_cast<int>(wx), static_cast<int>(wy),
                static_cast<int>(sx), static_cast<int>(sy),
                static_cast<int>(px), static_cast<int>(py),
                static_cast<int>(dx), static_cast<int>(dy),
                static_cast<int>(nx), static_cast<int>(ny),
                static_cast<int>(groups_x), static_cast<int>(groups_y));
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
