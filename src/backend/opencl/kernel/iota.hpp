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
#include <common/half.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/iota.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>
#include <string>

namespace opencl {
namespace kernel {
// Kernel Launch Config Values
static const int IOTA_TX = 32;
static const int IOTA_TY = 8;
static const int TILEX   = 512;
static const int TILEY   = 32;

template<typename T>
void iota(Param out, const af::dim4& sdims) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::Kernel;
    using cl::KernelFunctor;
    using cl::NDRange;
    using cl::Program;
    using std::string;

    std::string refName =
        std::string("iota_kernel_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName();
        options << getTypeBuildDefinition<T>();

        const char* ker_strs[] = {iota_cl};
        const int ker_lens[]   = {iota_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "iota_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto iotaOp =
        KernelFunctor<Buffer, const KParam, const int, const int, const int,
                      const int, const int, const int>(*entry.ker);

    NDRange local(IOTA_TX, IOTA_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                   local[1] * blocksPerMatY * out.info.dims[3], 1);

    iotaOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           sdims[0], sdims[1], sdims[2], sdims[3], blocksPerMatX,
           blocksPerMatY);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
