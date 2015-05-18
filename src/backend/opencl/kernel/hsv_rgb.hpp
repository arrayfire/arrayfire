/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once
#include <kernel_headers/hsv_rgb.hpp>
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

template<typename T, bool isHSV2RGB>
void hsv2rgb_convert(Param out, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  hrProgs;
        static std::map<int, Kernel*> hrKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();

                if(isHSV2RGB) options << " -D isHSV2RGB";

                if (std::is_same<T, double>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, hsv_rgb_cl, hsv_rgb_cl_len, options.str());
                hrProgs[device]   = new Program(prog);
                hrKernels[device] = new Kernel(*hrProgs[device], "convert");
            });

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(in.info.dims[0], THREADS_X);
        int blk_y = divup(in.info.dims[1], THREADS_Y);

        // all images are three channels, so batch
        // parameter would be along 4th dimension
        NDRange global(blk_x * in.info.dims[3] * THREADS_X, blk_y * THREADS_Y);

        auto hsvrgbOp = make_kernel<Buffer, KParam, Buffer, KParam, int> (*hrKernels[device]);

        hsvrgbOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
