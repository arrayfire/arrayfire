/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/convolve_separable.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <memory.hpp>

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

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_SCONV_FILTER_LEN = 31;

template<typename T, typename accType, int conv_dim, bool expand, int fLen>
void convolve2(Param out, const Param signal, const Param filter)
{
    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>   convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                const size_t C0_SIZE  = (THREADS_X+2*(fLen-1))* THREADS_Y;
                const size_t C1_SIZE  = (THREADS_Y+2*(fLen-1))* THREADS_X;

                size_t locSize = (conv_dim==0 ? C0_SIZE : C1_SIZE);

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<accType>::getName()
                            << " -D CONV_DIM="<< conv_dim
                            << " -D EXPAND="<< expand
                            << " -D FLEN="<< fLen
                            << " -D LOCAL_MEM_SIZE="<<locSize;
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, convolve_separable_cl, convolve_separable_cl_len, options.str());
                    convProgs[device]   = new Program(prog);
                    convKernels[device] = new Kernel(*convProgs[device], "convolve");
                });

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam, Buffer,
                                  int, int>(*convKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(out.info.dims[0], THREADS_X);
        int blk_y = divup(out.info.dims[1], THREADS_Y);

        NDRange global(blk_x*signal.info.dims[2]*THREADS_X,
                       blk_y*signal.info.dims[3]*THREADS_Y);

        cl::Buffer *mBuff = bufferAlloc(fLen*sizeof(accType));
        // FIX ME: if the filter array is strided, direct might cause issues
        getQueue().enqueueCopyBuffer(*filter.data, *mBuff, 0, 0, fLen*sizeof(accType));

        convOp(EnqueueArgs(getQueue(), global, local),
               *out.data, out.info, *signal.data, signal.info, *mBuff, blk_x, blk_y);

        bufferFree(mBuff);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
