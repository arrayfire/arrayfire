/*******************************************************
 * Copyright (c) 2017, ArrayFire
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
#include <kernel_headers/anisotropic_diffusion.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>

namespace opencl {
namespace kernel {
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int YDIM_LOAD = 2 * THREADS_X / THREADS_Y;

template<typename T, bool isMCDE>
void anisotropicDiffusion(Param inout, const float dt, const float mct,
                          const int fluxFnCode) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::Kernel;
    using cl::KernelFunctor;
    using cl::NDRange;
    using cl::Program;

    std::string kerKeyStr = std::string("anisotropic_diffusion_") +
                            std::string(dtype_traits<T>::getName()) + "_" +
                            std::to_string(isMCDE) + "_" +
                            std::to_string(fluxFnCode);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, kerKeyStr);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D SHRD_MEM_HEIGHT=" << (THREADS_Y * YDIM_LOAD + 2)
                << " -D SHRD_MEM_WIDTH=" << (THREADS_X + 2)
                << " -D IS_MCDE=" << isMCDE << " -D FLUX_FN=" << fluxFnCode
                << " -D YDIM_LOAD=" << YDIM_LOAD;
        options << getTypeBuildDefinition<T>();

        const char *ker_strs[] = {anisotropic_diffusion_cl};
        const int ker_lens[]   = {anisotropic_diffusion_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "diffUpdate");
        addKernelToCache(device, kerKeyStr, entry);
    }

    auto diffUpdateOp =
        KernelFunctor<Buffer, KParam, float, float, unsigned, unsigned>(
            *entry.ker);

    NDRange threads(THREADS_X, THREADS_Y, 1);

    int blkX = divup(inout.info.dims[0], threads[0]);
    int blkY = divup(inout.info.dims[1], threads[1] * YDIM_LOAD);

    NDRange global(threads[0] * blkX * inout.info.dims[2],
                   threads[1] * blkY * inout.info.dims[3], 1);

    diffUpdateOp(EnqueueArgs(getQueue(), global, threads), *inout.data,
                 inout.info, dt, mct, blkX, blkY);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
