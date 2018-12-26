/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
#include <kernel_headers/pad_array_borders.hpp>
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
static const int PADB_THREADS_X = 16;
static const int PADB_THREADS_Y = 16;

template<typename T, af_border_type BType>
void padBorders(Param out, const Param in, dim4 const& lBPadding) {
    std::string refName = std::string("padBorders_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(BType);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D AF_BORDER_TYPE=" << BType
                << " -D AF_PAD_SYM=" << AF_PAD_SYM
                << " -D AF_PAD_CLAMP_TO_EDGE=" << AF_PAD_CLAMP_TO_EDGE;
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {pad_array_borders_cl};
        const int ker_lens[]   = {pad_array_borders_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "padBorders");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(PADB_THREADS_X, PADB_THREADS_Y);

    int blk_x = divup(out.info.dims[0], local[0]);
    int blk_y = divup(out.info.dims[1], local[1]);

    NDRange global(blk_x * out.info.dims[2] * local[0],
                   blk_y * out.info.dims[3] * local[1]);

    auto padOP =
        KernelFunctor<Buffer, KParam, Buffer, KParam, unsigned, unsigned,
                      unsigned, unsigned, int, int>(*entry.ker);

    padOP(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
          in.info, lBPadding[0], lBPadding[1], lBPadding[2], lBPadding[3],
          blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
