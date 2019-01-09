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
#include <kernel_headers/nonmax_suppression.hpp>
#include <kernel_headers/trace_edge.hpp>
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
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template <typename T>
void nonMaxSuppression(Param output, const Param magnitude, const Param dx,
                       const Param dy) {
    std::string refName = std::string("non_max_suppression_") +
                          std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D SHRD_MEM_HEIGHT=" << (THREADS_X + 2)
                << " -D SHRD_MEM_WIDTH=" << (THREADS_Y + 2)
                << " -D NON_MAX_SUPPRESSION";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {nonmax_suppression_cl};
        const int ker_lens[]   = {nonmax_suppression_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "nonMaxSuppressionKernel");
        addKernelToCache(device, refName, entry);
    }

    auto nonMaxOp =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const Buffer, const KParam, const Buffer, const KParam,
                      const unsigned, const unsigned>(*entry.ker);

    NDRange threads(kernel::THREADS_X, kernel::THREADS_Y, 1);

    // Launch only threads to process non-border pixels
    int blk_x = divup(magnitude.info.dims[0] - 2, threads[0]);
    int blk_y = divup(magnitude.info.dims[1] - 2, threads[1]);

    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * magnitude.info.dims[2] * threads[0],
                   blk_y * magnitude.info.dims[3] * threads[1], 1);

    nonMaxOp(EnqueueArgs(getQueue(), global, threads), *output.data,
             output.info, *magnitude.data, magnitude.info, *dx.data, dx.info,
             *dy.data, dy.info, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void initEdgeOut(Param output, const Param strong, const Param weak) {
    std::string refName =
        std::string("init_edge_out_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D INIT_EDGE_OUT";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {trace_edge_cl};
        const int ker_lens[]   = {trace_edge_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "initEdgeOutKernel");
        addKernelToCache(device, refName, entry);
    }

    auto initOp = KernelFunctor<Buffer, const KParam, const Buffer,
                                const KParam, const Buffer, const KParam,
                                const unsigned, const unsigned>(*entry.ker);

    NDRange threads(kernel::THREADS_X, kernel::THREADS_Y, 1);

    // Launch only threads to process non-border pixels
    int blk_x = divup(strong.info.dims[0] - 2, threads[0]);
    int blk_y = divup(strong.info.dims[1] - 2, threads[1]);

    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * strong.info.dims[2] * threads[0],
                   blk_y * strong.info.dims[3] * threads[1], 1);

    initOp(EnqueueArgs(getQueue(), global, threads), *output.data, output.info,
           *strong.data, strong.info, *weak.data, weak.info, blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void suppressLeftOver(Param output) {
    std::string refName = std::string("suppress_left_over_") +
                          std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D SUPPRESS_LEFT_OVER";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {trace_edge_cl};
        const int ker_lens[]   = {trace_edge_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "suppressLeftOverKernel");
        addKernelToCache(device, refName, entry);
    }

    auto finalOp =
        KernelFunctor<Buffer, const KParam, const unsigned, const unsigned>(
            *entry.ker);

    NDRange threads(kernel::THREADS_X, kernel::THREADS_Y, 1);

    // Launch only threads to process non-border pixels
    int blk_x = divup(output.info.dims[0] - 2, threads[0]);
    int blk_y = divup(output.info.dims[1] - 2, threads[1]);

    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * output.info.dims[2] * threads[0],
                   blk_y * output.info.dims[3] * threads[1], 1);

    finalOp(EnqueueArgs(getQueue(), global, threads), *output.data, output.info,
            blk_x, blk_y);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void edgeTrackingHysteresis(Param output, const Param strong,
                            const Param weak) {
    std::string refName =
        std::string("edge_track_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D SHRD_MEM_HEIGHT=" << (THREADS_X + 2)
                << " -D SHRD_MEM_WIDTH=" << (THREADS_Y + 2)
                << " -D TOTAL_NUM_THREADS=" << (THREADS_X * THREADS_Y)
                << " -D EDGE_TRACER";
        if (std::is_same<T, double>::value) options << " -D USE_DOUBLE";

        const char *ker_strs[] = {trace_edge_cl};
        const int ker_lens[]   = {trace_edge_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "edgeTrackKernel");
        addKernelToCache(device, refName, entry);
    }

    NDRange threads(kernel::THREADS_X, kernel::THREADS_Y);

    // Launch only threads to process non-border pixels
    int blk_x = divup(weak.info.dims[0] - 2, threads[0]);
    int blk_y = divup(weak.info.dims[1] - 2, threads[1]);

    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * weak.info.dims[2] * threads[0],
                   blk_y * weak.info.dims[3] * threads[1], 1);

    auto edgeTraceOp = KernelFunctor<Buffer, const KParam, const unsigned,
                                     const unsigned, Buffer>(*entry.ker);

    initEdgeOut<T>(output, strong, weak);

    int notFinished        = 1;
    cl::Buffer *d_continue = bufferAlloc(sizeof(int));

    while (notFinished) {
        notFinished = 0;
        getQueue().enqueueWriteBuffer(*d_continue, CL_TRUE, 0, sizeof(int),
                                      &notFinished);

        edgeTraceOp(EnqueueArgs(getQueue(), global, threads), *output.data,
                    output.info, blk_x, blk_y, *d_continue);
        CL_DEBUG_FINISH(getQueue());

        getQueue().enqueueReadBuffer(*d_continue, CL_TRUE, 0, sizeof(int),
                                     &notFinished);
    }

    bufferFree(d_continue);

    suppressLeftOver<T>(output);
}
}  // namespace kernel
}  // namespace opencl
