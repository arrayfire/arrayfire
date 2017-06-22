/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/join.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
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
static const int TILEX = 256;
static const int TILEY = 32;

template<typename To, typename Ti, int dim>
void join(Param out, const Param in, const af::dim4 offset)
{
    std::string refName = std::string("join_kernel_") +
        std::string(dtype_traits<To>::getName()) +
        std::string(dtype_traits<Ti>::getName()) +
        std::to_string(dim);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Ti=" << dtype_traits<Ti>::getName()
                << " -D dim=" << dim;

        if (std::is_same<To, double>::value || std::is_same<To, cdouble>::value) {
            options << " -D USE_DOUBLE";
        } else if (std::is_same<Ti, double>::value || std::is_same<Ti, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {join_cl};
        const int   ker_lens[] = {join_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "join_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto joinOp = KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                                const int, const int, const int, const int,
                                const int, const int> (*entry.ker);

    NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(in.info.dims[0], TILEX);
    int blocksPerMatY = divup(in.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                   local[1] * blocksPerMatY * in.info.dims[3], 1);

    joinOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data, in.info,
           offset[0], offset[1], offset[2], offset[3], blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}
}
