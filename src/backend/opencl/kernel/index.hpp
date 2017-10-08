/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/index.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
static const int THREADS_X = 32;
static const int THREADS_Y =  8;

typedef struct {
    int  offs[4];
    int strds[4];
    char     isSeq[4];
} IndexKernelParam_t;

template<typename T>
void index(Param out, const Param in, const IndexKernelParam_t& p, Buffer *bPtr[4])
{
    std::string refName = std::string("indexKernel_") + std::string(dtype_traits<T>::getName());

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {index_cl};
        const int   ker_lens[] = {index_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "indexKernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    NDRange global(blk_x * out.info.dims[2] * THREADS_X, blk_y * out.info.dims[3] * THREADS_Y);

    auto indexOp = KernelFunctor<Buffer, KParam, Buffer, KParam, IndexKernelParam_t,
                                 Buffer, Buffer, Buffer, Buffer, int, int>(*entry.ker);

    indexOp(EnqueueArgs(getQueue(), global, local),
            *out.data, out.info, *in.data, in.info, p,
            *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3], blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}
}
