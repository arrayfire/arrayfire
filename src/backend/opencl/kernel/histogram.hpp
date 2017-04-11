/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/histogram.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Kernel;
using cl::KernelFunctor;

namespace opencl
{
namespace kernel
{
static const unsigned MAX_BINS  = 4000;
static const int THREADS_X =  256;
static const int THRD_LOAD =   16;

template<typename inType, typename outType, bool isLinear>
void histogram(Param out, const Param in, int nbins, float minval, float maxval)
{
    std::string refName = std::string("histogram_") +
        std::string(dtype_traits<inType>::getName()) +
        std::string(dtype_traits<inType>::getName()) +
        std::to_string(isLinear);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D inType=" << dtype_traits<inType>::getName()
            << " -D outType=" << dtype_traits<outType>::getName()
            << " -D THRD_LOAD=" << THRD_LOAD;
        if (isLinear)
            options << " -D IS_LINEAR";
        if (std::is_same<inType, double>::value ||
                std::is_same<inType, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {histogram_cl};
        const int   ker_lens[] = {histogram_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "histogram");

        addKernelToCache(device, refName, entry);
    }

    auto histogramOp = KernelFunctor< Buffer, KParam, Buffer, KParam, cl::LocalSpaceArg,
                                      int, int, float, float, int >(*entry.ker);

    int nElems = in.info.dims[0]*in.info.dims[1];
    int blk_x  = divup(nElems, THRD_LOAD*THREADS_X);
    int locSize = nbins * sizeof(outType);

    NDRange local(THREADS_X, 1);
    NDRange global(blk_x*in.info.dims[2]*THREADS_X, in.info.dims[3]);

    histogramOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, *in.data, in.info,
                cl::Local(locSize), nElems, nbins, minval, maxval, blk_x);

    CL_DEBUG_FINISH(getQueue());
}
}
}
