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
#include <common/dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/susan.hpp>
#include <memory.hpp>
#include <cache.hpp>
#include "config.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
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
           const unsigned in_off,
           const unsigned idim0, const unsigned idim1,
           const float t, const float g, const unsigned edge)
{
    std::string refName = std::string("susan_responses_") +
        std::string(dtype_traits<T>::getName()) + std::to_string(radius);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        const size_t LOCAL_MEM_SIZE = (SUSAN_THREADS_X+2*radius)*(SUSAN_THREADS_Y+2*radius);
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D LOCAL_MEM_SIZE=" << LOCAL_MEM_SIZE
                << " -D BLOCK_X="<< SUSAN_THREADS_X
                << " -D BLOCK_Y="<< SUSAN_THREADS_Y
                << " -D RADIUS="<< radius
                << " -D RESPONSE";
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {susan_cl};
        const int   ker_lens[] = {susan_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "susan_responses");

        addKernelToCache(device, refName, entry);
    }

    auto susanOp = KernelFunctor< Buffer, Buffer, unsigned, unsigned, unsigned,
                                  float, float, unsigned >(*entry.ker);

    NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
    NDRange global(divup(idim0-2*edge, local[0])*local[0], divup(idim1-2*edge, local[1])*local[1]);

    susanOp(EnqueueArgs(getQueue(), global, local), *out, *in, in_off, idim0, idim1, t, g, edge);
}

template<typename T>
unsigned nonMaximal(cl::Buffer* x_out, cl::Buffer* y_out, cl::Buffer* resp_out,
                    const unsigned idim0, const unsigned idim1, const cl::Buffer* resp_in,
                    const unsigned edge, const unsigned max_corners)
{
    unsigned corners_found = 0;

    std::string refName = std::string("non_maximal_") + std::string(dtype_traits<T>::getName());

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName() << " -D NONMAX";
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {susan_cl};
        const int   ker_lens[] = {susan_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "non_maximal");

        addKernelToCache(device, refName, entry);
    }

    cl::Buffer *d_corners_found = bufferAlloc(sizeof(unsigned));
    getQueue().enqueueWriteBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);

    auto nonMaximalOp = KernelFunctor< Buffer, Buffer, Buffer, Buffer, unsigned, unsigned, Buffer,
                                       unsigned, unsigned >(*entry.ker);

    NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
    NDRange global(divup(idim0-2*edge, local[0])*local[0], divup(idim1-2*edge, local[1])*local[1]);

    nonMaximalOp(EnqueueArgs(getQueue(), global, local),
                 *x_out, *y_out, *resp_out, *d_corners_found,
                 idim0, idim1, *resp_in, edge, max_corners);

    getQueue().enqueueReadBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned), &corners_found);
    bufferFree(d_corners_found);

    return corners_found;
}
}
}
