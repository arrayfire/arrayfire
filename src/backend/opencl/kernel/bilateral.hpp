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
#include <algorithm>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <af/opencl.h>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
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

template<typename inType, typename outType, bool isColor>
void bilateral(Param out, const Param in, float s_sigma, float c_sigma)
{
    std::string refName = std::string("bilateral_") +
        std::string(dtype_traits<inType>::getName()) +
        std::string(dtype_traits<outType>::getName()) +
        std::to_string(isColor);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D inType=" << dtype_traits<inType>::getName()
            << " -D outType=" << dtype_traits<outType>::getName();
        if (std::is_same<inType, double>::value ||
                std::is_same<inType, cdouble>::value) {
            options << " -D USE_DOUBLE";
        } else {
            options << " -D USE_NATIVE_EXP";
        }

        const char* ker_strs[] = {bilateral_cl};
        const int   ker_lens[] = {bilateral_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "bilateral");

        addKernelToCache(device, refName, entry);
    }

    auto bilateralOp = KernelFunctor< Buffer, KParam, Buffer, KParam, LocalSpaceArg, LocalSpaceArg,
                                      float, float, int, int, int >(*entry.ker);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    NDRange global(blk_x*in.info.dims[2]*THREADS_X, blk_y*in.info.dims[3]*THREADS_Y);

    // calculate local memory size
    int radius = (int)std::max(s_sigma * 1.5f, 1.f);
    int num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    int num_gauss_elems   = (2*radius+1)*(2*radius+1);
    size_t localMemSize   = (num_shrd_elems + num_gauss_elems)*sizeof(outType);
    size_t MaxLocalSize   = getDevice(getActiveDeviceId()).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    if (localMemSize>MaxLocalSize) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOpenCL Bilateral filter doesn't support %f spatial sigma\n", s_sigma);
        OPENCL_NOT_SUPPORTED(errMessage);
    }

    bilateralOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, *in.data, in.info,
                cl::Local(num_shrd_elems*sizeof(outType)),
                cl::Local(num_gauss_elems*sizeof(outType)),
                s_sigma, c_sigma, num_shrd_elems, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}
}
