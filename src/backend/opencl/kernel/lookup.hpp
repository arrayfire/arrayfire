/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/lookup.hpp>
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
static const int THREADS_Y = 8;

template<typename in_t, typename idx_t, unsigned dim>
void lookup(Param out, const Param in, const Param indices)
{
    std::string refName = std::string("lookupND_") +
        std::string(dtype_traits<in_t>::getName()) +
        std::string(dtype_traits<idx_t>::getName()) +
        std::to_string(dim);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D in_t=" << dtype_traits<in_t>::getName()
                << " -D idx_t=" << dtype_traits<idx_t>::getName()
                << " -D DIM=" <<dim;

        if (std::is_same<in_t, double>::value ||
                std::is_same<in_t, cdouble>::value ||
                std::is_same<idx_t, double>::value) {
            options << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {lookup_cl};
        const int   ker_lens[] = {lookup_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "lookupND");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    NDRange global(blk_x * out.info.dims[2] * THREADS_X, blk_y * out.info.dims[3] * THREADS_Y);

    auto arrIdxOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam, int, int>(*entry.ker);

    arrIdxOp(EnqueueArgs(getQueue(), global, local),
             *out.data, out.info, *in.data, in.info, *indices.data, indices.info, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}
}
