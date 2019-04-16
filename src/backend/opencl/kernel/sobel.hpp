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
#include <kernel_headers/sobel.hpp>
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

template<typename Ti, typename To, unsigned ker_size>
void sobel(Param dx, Param dy, const Param in) {
    std::string refName =
        std::string("sobel3x3_") + std::string(dtype_traits<Ti>::getName()) +
        std::string(dtype_traits<To>::getName()) + std::to_string(ker_size);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D Ti=" << dtype_traits<Ti>::getName()
                << " -D To=" << dtype_traits<To>::getName()
                << " -D KER_SIZE=" << ker_size;
        if (std::is_same<Ti, double>::value) options << " -D USE_DOUBLE";

        const char* ker_strs[] = {sobel_cl};
        const int ker_lens[]   = {sobel_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "sobel3x3");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                   blk_y * in.info.dims[3] * THREADS_Y);

    auto sobelOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                                 cl::LocalSpaceArg, int, int>(*entry.ker);

    size_t loc_size =
        (THREADS_X + ker_size - 1) * (THREADS_Y + ker_size - 1) * sizeof(Ti);

    sobelOp(EnqueueArgs(getQueue(), global, local), *dx.data, dx.info, *dy.data,
            dy.info, *in.data, in.info, cl::Local(loc_size), blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
