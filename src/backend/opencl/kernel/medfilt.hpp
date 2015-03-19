/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/medfilt.hpp>
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

static const dim_type MAX_MEDFILTER_LEN = 15;

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

template<typename T, af_pad_type pad, unsigned w_len, unsigned w_wid>
void medfilt(Param out, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  mfProgs;
        static std::map<int, Kernel*> mfKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                const dim_type ARR_SIZE = w_len * (w_wid-w_wid/2);

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D pad="<< pad
                        << " -D AF_ZERO="<< AF_ZERO
                        << " -D AF_SYMMETRIC="<< AF_SYMMETRIC
                        << " -D ARR_SIZE="<< ARR_SIZE
                        << " -D w_len="<< w_len
                        << " -D w_wid=" << w_wid;
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, medfilt_cl, medfilt_cl_len, options.str());
                mfProgs[device]   = new Program(prog);
                mfKernels[device] = new Kernel(*mfProgs[device], "medfilt");
            });

        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(in.info.dims[0], THREADS_X);
        dim_type blk_y = divup(in.info.dims[1], THREADS_Y);

        NDRange global(blk_x * in.info.dims[2] * in.info.dims[3] * THREADS_X,
                       blk_y * THREADS_Y);

        auto medfiltOp = make_kernel<Buffer, KParam,
                                     Buffer, KParam,
                                     cl::LocalSpaceArg,
                                     dim_type> (*mfKernels[device]);

        size_t loc_size = (THREADS_X+w_len-1)*(THREADS_Y+w_wid-1)*sizeof(T);

        medfiltOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info, cl::Local(loc_size), blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
