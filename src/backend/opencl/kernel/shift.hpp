/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/shift.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <cassert>

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
// Kernel Launch Config Values
static const int TX = 32;
static const int TY = 8;
static const int TILEX = 128;
static const int TILEY = 32;

template<typename T>
void shift(Param out, const Param in, const int *sdims)
{
    std::string refName = std::string("shift_kernel_") + std::string(dtype_traits<T>::getName());

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {shift_cl};
        const int   ker_lens[] = {shift_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "shift_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto shiftOp = KernelFunctor< Buffer, const Buffer, const KParam, const KParam,
                                  const int, const int, const int, const int,
                                  const int, const int> (*entry.ker);

    NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    int sdims_[4];
    // Need to do this because we are mapping output to input in the kernel
    for(int i = 0; i < 4; i++) {
        // sdims_[i] will always be positive and always [0, oDims[i]].
        // Negative shifts are converted to position by going the other way round
        sdims_[i] = -(sdims[i] % (int)out.info.dims[i]) + out.info.dims[i] * (sdims[i] > 0);
        assert(sdims_[i] >= 0 && sdims_[i] <= out.info.dims[i]);
    }

    shiftOp(EnqueueArgs(getQueue(), global, local),
            *out.data, *in.data, out.info, in.info,
            sdims_[0], sdims_[1], sdims_[2], sdims_[3],
            blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}
}
