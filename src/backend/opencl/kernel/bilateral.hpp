/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/bilateral.hpp>
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

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

template<typename inType, typename outType, bool isColor>
void bilateral(Param out, const Param in, float s_sigma, float c_sigma)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  bilProgs;
        static std::map<int, Kernel*> bilKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D inType=" << dtype_traits<inType>::getName()
                            << " -D outType=" << dtype_traits<outType>::getName();
                    if (std::is_same<inType, double>::value ||
                        std::is_same<inType, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, bilateral_cl, bilateral_cl_len, options.str());
                    bilProgs[device] = new Program(prog);

                    bilKernels[device] = new Kernel(*bilProgs[device], "bilateral");
                });

        auto bilateralOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       LocalSpaceArg,
                                       LocalSpaceArg,
                                       float, float,
                                       dim_type, dim_type
                                      >(*bilKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(in.info.dims[0], THREADS_X);
        dim_type blk_y = divup(in.info.dims[1], THREADS_Y);
        dim_type bCount= blk_x * (isColor ? in.info.dims[3] : in.info.dims[2] * in.info.dims[3]);

        NDRange global(bCount*THREADS_X, blk_y*THREADS_Y);

        // calculate local memory size
        dim_type radius = (dim_type)std::max(s_sigma * 1.5f, 1.f);
        dim_type num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
        dim_type num_gauss_elems   = (2*radius+1)*(2*radius+1);

        bilateralOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info,
                    cl::Local(num_shrd_elems*sizeof(outType)),
                    cl::Local(num_gauss_elems*sizeof(outType)),
                    s_sigma, c_sigma, num_shrd_elems, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
