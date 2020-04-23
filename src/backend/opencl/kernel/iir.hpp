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
#include <kernel_headers/iir.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <string>

using af::scalar_to_option;
using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
template<typename T, bool batch_a>
void iir(Param y, Param c, Param a) {
    // FIXME: This is a temporary fix. Ideally the local memory should be
    // allocted outside
    static const int MAX_A_SIZE = (1024 * sizeof(double)) / sizeof(T);

    std::string refName = std::string("iir_kernel_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(batch_a);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D MAX_A_SIZE=" << MAX_A_SIZE << " -D BATCH_A=" << batch_a
                << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")"
                << " -D T=" << dtype_traits<T>::getName();

        options << getTypeBuildDefinition<T>();

        const char* ker_strs[] = {iir_cl};
        const int ker_lens[]   = {iir_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "iir_kernel");

        addKernelToCache(device, refName, entry);
    }

    const int groups_y = y.info.dims[1];
    const int groups_x = y.info.dims[2];

    int threads = 256;
    while (threads > (int)y.info.dims[0] && threads > 32) threads /= 2;

    NDRange local(threads, 1);
    NDRange global(groups_x * local[0], groups_y * y.info.dims[3] * local[1]);

    auto iirOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam, int>(
            *entry.ker);

    try {
        iirOp(EnqueueArgs(getQueue(), global, local), *y.data, y.info, *c.data,
              c.info, *a.data, a.info, groups_y);
    } catch (cl::Error& clerr) {
        AF_ERROR("Size of a too big for this datatype", AF_ERR_SIZE);
    }

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
