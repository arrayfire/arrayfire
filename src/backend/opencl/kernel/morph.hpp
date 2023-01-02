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
#include <common/Binary.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/morph.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void morph(Param out, const Param in, const Param mask, bool isDilation) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::make_unique;
    using std::string;
    using std::vector;

    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    ToNumStr<T> toNumStr;
    const T DefaultVal = isDilation ? common::Binary<T, af_max_t>::init()
                                    : common::Binary<T, af_min_t>::init();
    const int windLen  = mask.info.dims[0];
    const int SeLength = (windLen <= 10 ? windLen : 0);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(isDilation),
        TemplateArg(SeLength),
    };
    vector<string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(isDilation),
        DefineValue(SeLength),
        DefineKeyValue(init, toNumStr(DefaultVal)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto morphOp = common::getKernel("morph", {{morph_cl_src}}, targs, options);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    NDRange global(blk_x * THREADS_X * in.info.dims[2],
                   blk_y * THREADS_Y * in.info.dims[3]);

    // copy mask/filter to read-only memory
    auto seBytes = windLen * windLen * sizeof(T);
    auto mBuff =
        make_unique<cl::Buffer>(getContext(), CL_MEM_READ_ONLY, seBytes);
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

template<typename T>
void morph3d(Param out, const Param in, const Param mask, bool isDilation) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::make_unique;
    using std::string;
    using std::vector;

    constexpr int CUBE_X = 8;
    constexpr int CUBE_Y = 8;
    constexpr int CUBE_Z = 4;

    ToNumStr<T> toNumStr;
    const T DefaultVal = isDilation ? common::Binary<T, af_max_t>::init()
                                    : common::Binary<T, af_min_t>::init();
    const int SeLength = mask.info.dims[0];

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(isDilation),
        TemplateArg(SeLength),
    };
    vector<string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(isDilation),
        DefineValue(SeLength),
        DefineKeyValue(init, toNumStr(DefaultVal)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto morphOp =
        common::getKernel("morph3d", {{morph_cl_src}}, targs, options);

    NDRange local(CUBE_X, CUBE_Y, CUBE_Z);

    int blk_x = divup(in.info.dims[0], CUBE_X);
    int blk_y = divup(in.info.dims[1], CUBE_Y);
    int blk_z = divup(in.info.dims[2], CUBE_Z);

    NDRange global(blk_x * CUBE_X * in.info.dims[3], blk_y * CUBE_Y,
                   blk_z * CUBE_Z);

    cl_int seBytes = sizeof(T) * SeLength * SeLength * SeLength;
    auto mBuff =
        make_unique<cl::Buffer>(getContext(), CL_MEM_READ_ONLY, seBytes);
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
}  // namespace arrayfire
