/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/meanshift.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <algorithm>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, bool is_color>
void meanshift(Param out, const Param in,
        const float spatialSigma, const float chromaticSigma, const uint numIters)
{
    typedef typename std::conditional< std::is_same<T, double>::value, double, float >::type AccType;

    std::string refName = std::string("meanshift_") +
        std::string(dtype_traits<T>::getName()) + std::to_string(is_color);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D AccType=" << dtype_traits<AccType>::getName()
                << " -D MAX_CHANNELS=" << (is_color ? 3 : 1);
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {meanshift_cl};
        const int   ker_lens[] = {meanshift_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "meanshift");

        addKernelToCache(device, refName, entry);
    }

    auto meanshiftOp = KernelFunctor<Buffer, KParam, Buffer, KParam,
                                    int, float, unsigned, int, int >(*entry.ker);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    const int bCount   = (is_color ? 1 : in.info.dims[2]);

    NDRange global(bCount*blk_x*THREADS_X, in.info.dims[3]*blk_y*THREADS_Y);

    // clamp spatical and chromatic sigma's
    int radius   = std::max( (int)(spatialSigma * 1.5f), 1 );

    const float cvar = chromaticSigma*chromaticSigma;

    meanshiftOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, *in.data, in.info,
                radius, cvar, numIters, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}
}
