/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <program.hpp>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/susan.hpp>
#include <memory.hpp>
#include <map>
#include "config.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;

namespace opencl
{

namespace kernel
{

static const unsigned THREADS_PER_BLOCK = 256;
static const unsigned SUSAN_THREADS_X = 16;
static const unsigned SUSAN_THREADS_Y = 16;

template<typename T, unsigned radius>
void susan(cl::Buffer* out, const cl::Buffer* in,
           const unsigned idim0, const unsigned idim1,
           const float t, const float g, const unsigned edge)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> suProg;
        static std::map<int, Kernel*>  suKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                const size_t LOCAL_MEM_SIZE = (SUSAN_THREADS_X+2*radius)*(SUSAN_THREADS_Y+2*radius);
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D LOCAL_MEM_SIZE=" << LOCAL_MEM_SIZE
                        << " -D BLOCK_X="<< SUSAN_THREADS_X
                        << " -D BLOCK_Y="<< SUSAN_THREADS_Y
                        << " -D RADIUS="<< radius
                        << " -D RESPONSE";

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, susan_cl, susan_cl_len, options.str());
                suProg[device]   = new Program(prog);
                suKernel[device] = new Kernel(*suProg[device], "susan_responses");
            });

        auto susanOp = make_kernel<Buffer, Buffer,
                                   unsigned, unsigned,
                                   float, float, unsigned>(*suKernel[device]);

        NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
        NDRange global(divup(idim0-2*edge, local[0])*local[0],
                       divup(idim1-2*edge, local[1])*local[1]);

        susanOp(EnqueueArgs(getQueue(), global, local), *out, *in, idim0, idim1, t, g, edge);

    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T>
unsigned nonMaximal(cl::Buffer* x_out, cl::Buffer* y_out, cl::Buffer* resp_out,
                    const unsigned idim0, const unsigned idim1, const cl::Buffer* resp_in,
                    const unsigned edge, const unsigned max_corners)
{
    unsigned corners_found = 0;
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> nmProg;
        static std::map<int, Kernel*>  nmKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D NONMAX";

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, susan_cl, susan_cl_len, options.str());
                nmProg[device]   = new Program(prog);
                nmKernel[device] = new Kernel(*nmProg[device], "non_maximal");
            });

        cl::Buffer *d_corners_found = bufferAlloc(sizeof(unsigned));
        getQueue().enqueueWriteBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);

        auto nonMaximalOp = make_kernel<Buffer, Buffer, Buffer, Buffer,
                                        unsigned, unsigned, Buffer,
                                        unsigned, unsigned>(*nmKernel[device]);

        NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
        NDRange global(divup(idim0-2*edge, local[0])*local[0],
                       divup(idim1-2*edge, local[1])*local[1]);

        nonMaximalOp(EnqueueArgs(getQueue(), global, local),
                     *x_out, *y_out, *resp_out, *d_corners_found,
                     idim0, idim1, *resp_in, edge, max_corners);

        getQueue().enqueueReadBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);
        bufferFree(d_corners_found);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
    return corners_found;
}

}

}
