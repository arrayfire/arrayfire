/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/sobel.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{

namespace kernel
{

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename Ti, typename To, unsigned ker_size>
void sobel(Param dx, Param dy, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  sobProgs;
        static std::map<int, Kernel*> sobKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D Ti=" << dtype_traits<Ti>::getName()
                        << " -D To=" << dtype_traits<To>::getName()
                        << " -D KER_SIZE="<< ker_size;
                if (std::is_same<Ti, double>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, sobel_cl, sobel_cl_len, options.str());
                sobProgs[device]   = new Program(prog);
                sobKernels[device] = new Kernel(*sobProgs[device], "sobel3x3");
            });

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(in.info.dims[0], THREADS_X);
        int blk_y = divup(in.info.dims[1], THREADS_Y);

        NDRange global(blk_x * in.info.dims[2] * THREADS_X,
                       blk_y * in.info.dims[3] * THREADS_Y);

        auto sobelOp = make_kernel<Buffer, KParam,
                                   Buffer, KParam,
                                   Buffer, KParam,
                                   cl::LocalSpaceArg,
                                   int, int> (*sobKernels[device]);

        size_t loc_size = (THREADS_X+ker_size-1)*(THREADS_Y+ker_size-1)*sizeof(Ti);

        sobelOp(EnqueueArgs(getQueue(), global, local),
                    *dx.data, dx.info, *dy.data, dy.info,
                    *in.data, in.info, cl::Local(loc_size), blk_x, blk_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
