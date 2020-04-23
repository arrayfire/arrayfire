/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/join.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <map>
#include <mutex>
#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
// Kernel Launch Config Values
static const int TX    = 32;
static const int TY    = 8;
static const int TILEX = 256;
static const int TILEY = 32;

template<typename T>
void join(Param out, const Param in, dim_t dim, const af::dim4 offset) {
    std::string refName =
        std::string("join_kernel_") + std::string(dtype_traits<T>::getName()) +
        std::string(dtype_traits<T>::getName()) + std::to_string(dim);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << getTypeBuildDefinition<T>();

        const char* ker_strs[] = {join_cl};
        const int ker_lens[]   = {join_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "join_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto joinOp = KernelFunctor<Buffer, const KParam, const Buffer,
                                const KParam, const int, const int, const int,
                                const int, const int, const int>(*entry.ker);

    NDRange local(TX, TY, 1);

    int blocksPerMatX = divup(in.info.dims[0], TILEX);
    int blocksPerMatY = divup(in.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                   local[1] * blocksPerMatY * in.info.dims[3], 1);

    joinOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *in.data, in.info, offset[0], offset[1], offset[2], offset[3],
           blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
