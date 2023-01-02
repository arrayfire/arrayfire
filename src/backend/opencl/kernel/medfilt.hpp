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
#include <kernel_headers/medfilt1.hpp>
#include <kernel_headers/medfilt2.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr int MAX_MEDFILTER2_LEN = 15;
constexpr int MAX_MEDFILTER1_LEN = 121;

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

template<typename T>
void medfilt1(Param out, const Param in, const unsigned w_wid,
              const af_border_type pad) {
    const int ARR_SIZE = (w_wid - w_wid / 2) + 1;
    size_t loc_size    = (THREADS_X + w_wid - 1) * sizeof(T);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(pad),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(pad, static_cast<int>(pad)),
        DefineKeyValue(AF_PAD_ZERO, static_cast<int>(AF_PAD_ZERO)),
        DefineKeyValue(AF_PAD_SYM, static_cast<int>(AF_PAD_SYM)),
        DefineValue(ARR_SIZE),
        DefineValue(w_wid),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto medfiltOp =
        common::getKernel("medfilt1", {{medfilt1_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, 1, 1);

    int blk_x = divup(in.info.dims[0], THREADS_X);

    cl::NDRange global(blk_x * in.info.dims[1] * THREADS_X, in.info.dims[2],
                       in.info.dims[3]);

    medfiltOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, cl::Local(loc_size), blk_x);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void medfilt2(Param out, const Param in, const af_border_type pad,
              const unsigned w_len, const unsigned w_wid) {
    const int ARR_SIZE = w_len * (w_wid - w_wid / 2);
    const size_t loc_size =
        (THREADS_X + w_len - 1) * (THREADS_Y + w_wid - 1) * sizeof(T);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(pad),
        TemplateArg(w_len),
        TemplateArg(w_wid),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(pad, static_cast<int>(pad)),
        DefineKeyValue(AF_PAD_ZERO, static_cast<int>(AF_PAD_ZERO)),
        DefineKeyValue(AF_PAD_SYM, static_cast<int>(AF_PAD_SYM)),
        DefineValue(ARR_SIZE),
        DefineValue(w_wid),
        DefineValue(w_len),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto medfiltOp =
        common::getKernel("medfilt2", {{medfilt2_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);

    medfiltOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, cl::Local(loc_size), blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
