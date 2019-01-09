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
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/matchTemplate.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template <typename inType, typename outType, af_match_type mType, bool needMean>
void matchTemplate(Param out, const Param srch, const Param tmplt) {
    std::string refName = std::string("matchTemplate_") +
                          std::string(dtype_traits<inType>::getName()) +
                          std::string(dtype_traits<outType>::getName()) +
                          std::to_string(mType) + std::to_string(needMean);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D inType=" << dtype_traits<inType>::getName()
                << " -D outType=" << dtype_traits<outType>::getName()
                << " -D MATCH_T=" << mType << " -D NEEDMEAN=" << needMean
                << " -D AF_SAD=" << AF_SAD << " -D AF_ZSAD=" << AF_ZSAD
                << " -D AF_LSAD=" << AF_LSAD << " -D AF_SSD=" << AF_SSD
                << " -D AF_ZSSD=" << AF_ZSSD << " -D AF_LSSD=" << AF_LSSD
                << " -D AF_NCC=" << AF_NCC << " -D AF_ZNCC=" << AF_ZNCC
                << " -D AF_SHD=" << AF_SHD;
        if (std::is_same<outType, double>::value) options << " -D USE_DOUBLE";

        const char* ker_strs[] = {matchTemplate_cl};
        const int ker_lens[]   = {matchTemplate_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "matchTemplate");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(srch.info.dims[0], THREADS_X);
    int blk_y = divup(srch.info.dims[1], THREADS_Y);

    NDRange global(blk_x * srch.info.dims[2] * THREADS_X,
                   blk_y * srch.info.dims[3] * THREADS_Y);

    auto matchImgOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam, int, int>(
            *entry.ker);

    matchImgOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *srch.data, srch.info, *tmplt.data, tmplt.info, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
