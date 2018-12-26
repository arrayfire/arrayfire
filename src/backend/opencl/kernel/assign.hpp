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
#include <kernel_headers/assign.hpp>
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
static const int THREADS_X = 32;
static const int THREADS_Y = 8;

typedef struct {
    int offs[4];
    int strds[4];
    char isSeq[4];
} AssignKernelParam_t;

template<typename T>
void assign(Param out, const Param in, const AssignKernelParam_t& p,
            Buffer* bPtr[4]) {
    std::string refName =
        std::string("assignKernel_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {assign_cl};
        const int ker_lens[]   = {assign_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "assignKernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                   blk_y * in.info.dims[3] * THREADS_Y);

    auto assignOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, AssignKernelParam_t,
                      Buffer, Buffer, Buffer, Buffer, int, int>(*entry.ker);

    assignOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, p, *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3],
             blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
