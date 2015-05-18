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
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Kernel;

namespace opencl
{

namespace kernel
{

static const unsigned MAX_BINS  = 4000;
static const int THREADS_X =  256;
static const int THRD_LOAD =   16;

template<typename inType, typename outType>
void histogram(Param out, const Param in, const Param minmax, int nbins)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> histProgs;
        static std::map<int, Kernel *> histKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D inType=" << dtype_traits<inType>::getName()
                            << " -D outType=" << dtype_traits<outType>::getName()
                            << " -D THRD_LOAD=" << THRD_LOAD;

                    if (std::is_same<inType, double>::value ||
                        std::is_same<inType, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, histogram_cl, histogram_cl_len, options.str());
                    histProgs[device]   = new Program(prog);
                    histKernels[device] = new Kernel(*histProgs[device], "histogram");
                });

        auto histogramOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                       Buffer, cl::LocalSpaceArg,
                                       int, int, int
                                      >(*histKernels[device]);

        int nElems = in.info.dims[0]*in.info.dims[1];
        int blk_x  = divup(nElems, THRD_LOAD*THREADS_X);
        int locSize = nbins * sizeof(outType);

        NDRange local(THREADS_X, 1);
        NDRange global(blk_x*in.info.dims[2]*THREADS_X, in.info.dims[3]);

        histogramOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, *in.data, in.info, *minmax.data,
                cl::Local(locSize), nElems, nbins, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
