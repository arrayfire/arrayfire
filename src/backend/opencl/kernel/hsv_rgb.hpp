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
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
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
    std::string refName = std::string("hsvrgb_convert_") +
        std::string(dtype_traits<T>::getName()) + std::to_string(isHSV2RGB);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();

        if(isHSV2RGB) options << " -D isHSV2RGB";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char* ker_strs[] = {hsv_rgb_cl};
        const int   ker_lens[] = {hsv_rgb_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "convert");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    // all images are three channels, so batch
    // parameter would be along 4th dimension
    NDRange global(blk_x * in.info.dims[3] * THREADS_X, blk_y * THREADS_Y);

    auto hsvrgbOp = KernelFunctor<Buffer, KParam, Buffer, KParam, int> (*entry.ker);

    hsvrgbOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data, in.info, blk_x);

    CL_DEBUG_FINISH(getQueue());
}
}
}
