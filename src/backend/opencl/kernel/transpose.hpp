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
#include <kernel_headers/transpose.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <string>

namespace opencl {
namespace kernel {
static const int TILE_DIM  = 32;
static const int THREADS_X = TILE_DIM;
static const int THREADS_Y = 256 / TILE_DIM;

template<typename T, bool conjugate, bool IS32MULTIPLE>
void transpose(Param out, const Param in, cl::CommandQueue queue) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::Kernel;
    using cl::KernelFunctor;
    using cl::NDRange;
    using cl::Program;
    using std::string;

    string refName =
        std::string("transpose_") + std::string(dtype_traits<T>::getName()) +
        std::to_string(conjugate) + std::to_string(IS32MULTIPLE);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D TILE_DIM=" << TILE_DIM << " -D THREADS_Y=" << THREADS_Y
                << " -D IS32MULTIPLE=" << IS32MULTIPLE
                << " -D DOCONJUGATE=" << (conjugate && af::iscplx<T>())
                << " -D T=" << dtype_traits<T>::getName();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        if (std::is_same<T, common::half>::value) options << " -D USE_HALF";

        const char* ker_strs[] = {transpose_cl};
        const int ker_lens[]   = {transpose_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "transpose");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], TILE_DIM);
    int blk_y = divup(in.info.dims[1], TILE_DIM);

    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * local[0] * in.info.dims[2],
                   blk_y * local[1] * in.info.dims[3]);

    auto transposeOp =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const int, const int>(*entry.ker);

    transposeOp(EnqueueArgs(queue, global, local), *out.data, out.info,
                *in.data, in.info, blk_x, blk_y);

    CL_DEBUG_FINISH(queue);
}
}  // namespace kernel
}  // namespace opencl
