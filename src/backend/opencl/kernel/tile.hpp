/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/tile.hpp>
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
// Kernel Launch Config Values
static const int TX = 32;
static const int TY = 8;
static const int TILEX = 512;
static const int TILEY = 32;

template<typename T>
void tile(Param out, const Param in)
{
    std::string refName = std::string("tile_kernel_") + std::string(dtype_traits<T>::getName());

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {tile_cl};
        const int   ker_lens[] = {tile_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "tile_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto tileOp = KernelFunctor< Buffer, const Buffer, const KParam, const KParam,
                                 const int, const int> (*entry.ker);

    NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    tileOp(EnqueueArgs(getQueue(), global, local),
            *out.data, *in.data, out.info, in.info, blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}
}
