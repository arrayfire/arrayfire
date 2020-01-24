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
#include <kernel_headers/morph.hpp>
#include <memory.hpp>
#include <ops.hpp>
#include <traits.hpp>
#include <type_util.hpp>

#include <memory>
#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::LocalSpaceArg;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const int CUBE_X = 8;
static const int CUBE_Y = 8;
static const int CUBE_Z = 4;

template<typename T, bool isDilation, int SeLength = 0>
void morph(Param out, const Param in, const Param mask, int windLen = 0) {
    ToNumStr<T> toNumStr;
    constexpr bool TypeIsDouble =
        (std::is_same<T, double>::value || std::is_same<T, cdouble>::value);
    const T DefaultVal =
        isDilation ? Binary<T, af_max_t>::init() : Binary<T, af_min_t>::init();

    static const std::string src(morph_cl, morph_cl_len);

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(isDilation),
        TemplateArg(SeLength),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(isDilation),
        DefineValue(SeLength),
        DefineKeyValue(init, toNumStr(DefaultVal)),
    };

    if (TypeIsDouble) { compileOpts.emplace_back(DefineKey(USE_DOUBLE)); }

    auto morphOp = common::findKernel("morph", {src}, tmpltArgs, compileOpts);

    windLen = (SeLength > 0 ? SeLength : windLen);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    NDRange global(blk_x * THREADS_X * in.info.dims[2],
                   blk_y * THREADS_Y * in.info.dims[3]);

    // copy mask/filter to read-only memory
    auto seBytes = windLen * windLen * sizeof(T);
    auto mBuff =
        std::make_unique<cl::Buffer>(getContext(), CL_MEM_READ_ONLY, seBytes);
    morphOp.copyToReadOnly(mBuff.get(), mask.data, seBytes);

    // calculate shared memory size
    const int padding =
        (windLen % 2 == 0 ? (windLen - 1) : (2 * (windLen / 2)));
    const int locLen  = THREADS_X + padding + 1;
    const int locSize = locLen * (THREADS_Y + padding);

    morphOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            *in.data, in.info, *mBuff, cl::Local(locSize * sizeof(T)), blk_x,
            blk_y, windLen);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, bool isDilation, int SeLength>
void morph3d(Param out, const Param in, const Param mask) {
    ToNumStr<T> toNumStr;
    constexpr bool TypeIsDouble =
        (std::is_same<T, double>::value || std::is_same<T, cdouble>::value);
    const T DefaultVal =
        isDilation ? Binary<T, af_max_t>::init() : Binary<T, af_min_t>::init();

    static const std::string src(morph_cl, morph_cl_len);

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(isDilation),
        TemplateArg(SeLength),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(isDilation),
        DefineValue(SeLength),
        DefineKeyValue(init, toNumStr(DefaultVal)),
    };

    if (TypeIsDouble) { compileOpts.emplace_back(DefineKey(USE_DOUBLE)); }

    auto morphOp = common::findKernel("morph3d", {src}, tmpltArgs, compileOpts);

    NDRange local(CUBE_X, CUBE_Y, CUBE_Z);

    int blk_x = divup(in.info.dims[0], CUBE_X);
    int blk_y = divup(in.info.dims[1], CUBE_Y);
    int blk_z = divup(in.info.dims[2], CUBE_Z);

    NDRange global(blk_x * CUBE_X * in.info.dims[3], blk_y * CUBE_Y,
                   blk_z * CUBE_Z);

    cl_int seBytes = sizeof(T) * SeLength * SeLength * SeLength;
    auto mBuff =
        std::make_unique<cl::Buffer>(getContext(), CL_MEM_READ_ONLY, seBytes);
    morphOp.copyToReadOnly(mBuff.get(), mask.data, seBytes);

    // calculate shared memory size
    const int padding =
        (SeLength % 2 == 0 ? (SeLength - 1) : (2 * (SeLength / 2)));
    const int locLen  = CUBE_X + padding + 1;
    const int locArea = locLen * (CUBE_Y + padding);
    const int locSize = locArea * (CUBE_Z + padding);

    morphOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            *in.data, in.info, *mBuff, cl::Local(locSize * sizeof(T)), blk_x);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
