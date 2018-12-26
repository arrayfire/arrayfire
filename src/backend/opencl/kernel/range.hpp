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
#include <kernel_headers/range.hpp>
#include <program.hpp>
#include <traits.hpp>
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
static const int RANGE_TX    = 32;
static const int RANGE_TY    = 8;
static const int RANGE_TILEX = 512;
static const int RANGE_TILEY = 32;

template<typename T>
void range(Param out, const int dim) {
    std::string refName =
        std::string("range_kernel_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {range_cl};
        const int ker_lens[]   = {range_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "range_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto rangeOp =
        KernelFunctor<Buffer, const KParam, const int, const int, const int>(
            *entry.ker);

    NDRange local(RANGE_TX, RANGE_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.info.dims[1], RANGE_TILEY);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    rangeOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, dim,
            blocksPerMatX, blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
