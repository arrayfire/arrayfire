/*******************************************************
 * Copyright (c) 2016, ArrayFire
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
#include <kernel_headers/moments.hpp>
#include <math.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <map>
#include <mutex>
#include <string>
#include "config.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int THREADS = 128;

///////////////////////////////////////////////////////////////////////////
// Wrapper functions
///////////////////////////////////////////////////////////////////////////
template <typename T>
void moments(Param out, const Param in, af_moment_type moment) {
    std::string ref_name = std::string("moments_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string("_") + std::to_string(out.info.dims[0]);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << " -D MOMENTS_SZ=" << out.info.dims[0];

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        Program prog;
        buildProgram(prog, moments_cl, moments_cl_len, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "moments_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    auto momentsp =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const int, const int>(*entry.ker);

    NDRange local(THREADS, 1, 1);
    NDRange global(in.info.dims[1] * local[0],
                   in.info.dims[2] * in.info.dims[3] * local[1]);

    bool pBatch = !(in.info.dims[2] == 1 && in.info.dims[3] == 1);

    momentsp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, (int)moment, (int)pBatch);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
