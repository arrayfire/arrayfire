/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/index.hpp>
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

static const int THREADS_X = 32;
static const int THREADS_Y =  8;

typedef struct {
    int  offs[4];
    int strds[4];
    char     isSeq[4];
} IndexKernelParam_t;

template<typename T>
void index(Param out, const Param in, const IndexKernelParam_t& p, Buffer *bPtr[4])
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  idxProgs;
        static std::map<int, Kernel*> idxKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
                }

                Program prog;
                buildProgram(prog, index_cl, index_cl_len, options.str());
                idxProgs[device]   = new Program(prog);
                idxKernels[device] = new Kernel(*idxProgs[device], "indexKernel");
                });

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(out.info.dims[0], THREADS_X);
        int blk_y = divup(out.info.dims[1], THREADS_Y);

        NDRange global(blk_x * out.info.dims[2] * THREADS_X,
                blk_y * out.info.dims[3] * THREADS_Y);

        auto indexOp = make_kernel<Buffer, KParam, Buffer, KParam, IndexKernelParam_t,
             Buffer, Buffer, Buffer, Buffer, int, int>(*idxKernels[device]);

        indexOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, *in.data, in.info, p,
                *bPtr[0], *bPtr[1], *bPtr[2], *bPtr[3], blk_x, blk_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
