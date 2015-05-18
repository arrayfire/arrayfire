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
#include <mutex>
#include <map>
#include <algorithm>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
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
void meanshift(Param out, const Param in, float s_sigma, float c_sigma, uint iter)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> msProgs;
        static std::map<int, Kernel*> msKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D MAX_CHANNELS=" << (is_color ? 3 : 1);
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, meanshift_cl, meanshift_cl_len, options.str());
                    msProgs[device]   = new Program(prog);
                    msKernels[device] = new Kernel(*msProgs[device], "meanshift");
                });

        auto meanshiftOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       LocalSpaceArg,
                                       int, float,
                                       int, float,
                                       unsigned, int, int
                                      >(*msKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(in.info.dims[0], THREADS_X);
        int blk_y = divup(in.info.dims[1], THREADS_Y);

        const int bCount   = (is_color ? 1 : in.info.dims[2]);
        const int channels = (is_color ? in.info.dims[2] : 1);

        NDRange global(bCount*blk_x*THREADS_X, in.info.dims[3]*blk_y*THREADS_Y);

        // clamp spatical and chromatic sigma's
        float space_ = std::min(11.5f, s_sigma);
        int radius   = std::max((int)(space_ * 1.5f), 1);
        int padding  = 2*radius+1;
        const float cvar = c_sigma*c_sigma;
        size_t loc_size  = channels*(local[0]+padding)*(local[1]+padding)*sizeof(T);

        meanshiftOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info,
                    cl::Local(loc_size), channels,
                    space_, radius, cvar, iter, blk_x, blk_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
