/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/identity.hpp>
#include <math.hpp>
#include <program.hpp>
#include <traits.hpp>
#include "config.hpp"

using af::scalar_to_option;
using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::ostringstream;
using std::string;

namespace opencl {
namespace kernel {
template<typename T>
static void identity(Param out) {
    std::string refName = std::string("identity_kernel") +
                          std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName() << " -D ONE=(T)("
                << scalar_to_option(scalar<T>(1)) << ")"
                << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")";
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {identity_cl};
        const int ker_lens[]   = {identity_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "identity_kernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(32, 8);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    auto identityOp = KernelFunctor<Buffer, const KParam, int, int>(*entry.ker);

    identityOp(EnqueueArgs(getQueue(), global, local), *(out.data), out.info,
               groups_x, groups_y);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
