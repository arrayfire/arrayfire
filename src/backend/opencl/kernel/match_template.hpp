/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/matchTemplate.hpp>
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

template<typename inType, typename outType, af_match_type mType, bool needMean>
void matchTemplate(Param out, const Param srch, const Param tmplt)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  mtProgs;
        static std::map<int, Kernel*> mtKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D inType="  << dtype_traits<inType>::getName()
                        << " -D outType=" << dtype_traits<outType>::getName()
                        << " -D MATCH_T=" << mType
                        << " -D NEEDMEAN="<< needMean
                        << " -D AF_SAD="  << AF_SAD
                        << " -D AF_ZSAD=" << AF_ZSAD
                        << " -D AF_LSAD=" << AF_LSAD
                        << " -D AF_SSD="  << AF_SSD
                        << " -D AF_ZSSD=" << AF_ZSSD
                        << " -D AF_LSSD=" << AF_LSSD
                        << " -D AF_NCC="  << AF_NCC
                        << " -D AF_ZNCC=" << AF_ZNCC
                        << " -D AF_SHD="  << AF_SHD;
                if (std::is_same<outType, double>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, matchTemplate_cl, matchTemplate_cl_len, options.str());
                mtProgs[device]   = new Program(prog);
                mtKernels[device] = new Kernel(*mtProgs[device], "matchTemplate");
            });

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(srch.info.dims[0], THREADS_X);
        int blk_y = divup(srch.info.dims[1], THREADS_Y);

        NDRange global(blk_x * srch.info.dims[2] * THREADS_X, blk_y * srch.info.dims[3] * THREADS_Y);

        auto matchImgOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       Buffer, KParam,
                                       int, int> (*mtKernels[device]);

        matchImgOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *srch.data, srch.info, *tmplt.data, tmplt.info, blk_x, blk_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
