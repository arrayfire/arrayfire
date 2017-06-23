/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/diff.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <dispatch.hpp>
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
static const int TX = 16;
static const int TY = 16;

template<typename T, unsigned dim, bool isDiff2>
void diff(Param out, const Param in, const unsigned indims)
{
    std::string refName = std::string("diff_kernel_") +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(dim) +
        std::to_string(isDiff2);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T="        << dtype_traits<T>::getName()
                << " -D DIM="      << dim
                << " -D isDiff2=" << isDiff2;
        if (std::is_same<T, double>::value ||
            std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {diff_cl};
        const int   ker_lens[] = {diff_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "diff_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto diffOp = KernelFunctor< Buffer, const Buffer, const KParam, const KParam,
                                 const int, const int, const int> (*entry.ker);

    NDRange local(TX, TY, 1);
    if(dim == 0 && indims == 1) {
        local = NDRange(TX * TY, 1, 1);
    }

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    const int oElem = out.info.dims[0] * out.info.dims[1] * out.info.dims[2] * out.info.dims[3];

    diffOp(EnqueueArgs(getQueue(), global, local),
           *out.data, *in.data, out.info, in.info, oElem, blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}
}
