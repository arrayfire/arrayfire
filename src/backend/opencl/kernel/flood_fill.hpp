/*******************************************************
 * Copyright (c) 2019, ArrayFire
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
#include <kernel_headers/flood_fill.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int TILE_DIM  = 16;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = THREADS / TILE_DIM;
constexpr int VALID     = 2;
constexpr int INVALID   = 1;
constexpr int ZERO      = 0;

template<typename T>
void initSeeds(Param out, const Param seedsx, const Param seedsy) {
    std::string refName = std::string("init_seeds_") +
                          std::string(dtype_traits<T>::getName());
    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D VALID=" << T(VALID)
                << " -D INIT_SEEDS";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {flood_fill_cl};
        const int ker_lens[]   = {flood_fill_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "init_seeds");
        addKernelToCache(device, refName, entry);
    }
    auto initSeedsOp = KernelFunctor<Buffer, const KParam,
                                     const Buffer, const KParam,
                                     const Buffer, const KParam>(*entry.ker);
    NDRange local(kernel::THREADS, 1, 1);
    NDRange global( divup(seedsx.info.dims[0], local[0]) * local[0], 1 , 1);

    initSeedsOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *seedsx.data, seedsx.info, *seedsy.data, seedsy.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void finalizeOutput(Param out, const T newValue) {
    std::string refName = std::string("finalize_output_") +
                          std::string(dtype_traits<T>::getName());
    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D VALID=" << T(VALID)
                << " -D ZERO=" << T(ZERO)
                << " -D FINALIZE_OUTPUT";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {flood_fill_cl};
        const int ker_lens[]   = {flood_fill_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "finalize_output");
        addKernelToCache(device, refName, entry);
    }

    auto finalizeOut = KernelFunctor<Buffer, const KParam, const T>(*entry.ker);

    NDRange local(kernel::THREADS_X, kernel::THREADS_Y, 1);
    NDRange global( divup(out.info.dims[0], local[0]) * local[0],
                    divup(out.info.dims[1], local[1]) * local[1] ,
                    1);
    finalizeOut(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info, newValue);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void floodFill(Param out, const Param image, const Param seedsx,
               const Param seedsy, const T newValue, const T lowValue,
               const T highValue, const af::connectivity nlookup) {
    constexpr int RADIUS = 1;
    UNUSED(nlookup);
    std::string refName = std::string("flood_step_") +
                          std::string(dtype_traits<T>::getName());
    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D RADIUS=" << RADIUS
                << " -D LMEM_WIDTH=" << (THREADS_X + 2 * RADIUS)
                << " -D LMEM_HEIGHT=" << (THREADS_Y + 2 * RADIUS)
                << " -D GROUP_SIZE=" << (THREADS_Y * THREADS_X)
                << " -D VALID=" << T(VALID)
                << " -D INVALID=" << T(INVALID)
                << " -D ZERO=" << T(ZERO)
                << " -D FLOOD_FILL_STEP";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {flood_fill_cl};
        const int ker_lens[]   = {flood_fill_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "flood_step");

        addKernelToCache(device, refName, entry);
    }
    auto floodStep = KernelFunctor<Buffer, const KParam,
                                   const Buffer, const KParam,
                                   const T, const T, Buffer>(*entry.ker);
    NDRange local(kernel::THREADS_X, kernel::THREADS_Y, 1);
    NDRange global( divup(out.info.dims[0], local[0]) * local[0],
                    divup(out.info.dims[1], local[1]) * local[1] ,
                    1);

    initSeeds<T>(out, seedsx, seedsy);

    int notFinished       = 1;
    cl::Buffer *dContinue = bufferAlloc(sizeof(int));

    while (notFinished) {
        notFinished = 0;
        getQueue().enqueueWriteBuffer(*dContinue, CL_TRUE, 0, sizeof(int),
                                      &notFinished);

        floodStep(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                  *image.data, image.info, lowValue, highValue, *dContinue);
        CL_DEBUG_FINISH(getQueue());

        getQueue().enqueueReadBuffer(*dContinue, CL_TRUE, 0, sizeof(int),
                                     &notFinished);
    }

    bufferFree(dContinue);

    finalizeOutput<T>(out, newValue);
}

}
}
