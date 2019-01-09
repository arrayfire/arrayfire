/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <kernel_headers/select.hpp>
#include <math.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <types.hpp>
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
static const uint DIMX  = 32;
static const uint DIMY  = 8;
static const int REPEAT = 64;

template <typename T, bool is_same>
void select_launcher(Param out, Param cond, Param a, Param b, int ndims) {
    std::string refName = std::string("select_kernel_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(is_same);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D is_same=" << is_same
                << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {select_cl};
        const int ker_lens[]   = {select_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "select_kernel");

        addKernelToCache(device, refName, entry);
    }

    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    NDRange local(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    NDRange global(groups_0 * out.info.dims[2] * local[0],
                   groups_1 * out.info.dims[3] * local[1]);

    auto selectOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                  KParam, Buffer, KParam, int, int>(*entry.ker);

    selectOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *cond.data, cond.info, *a.data, a.info, *b.data, b.info, groups_0,
             groups_1);
}

template <typename T>
void select(Param out, Param cond, Param a, Param b, int ndims) {
    bool is_same = true;
    for (int i = 0; i < 4; i++) {
        is_same &= (a.info.dims[i] == b.info.dims[i]);
    }

    if (is_same) {
        select_launcher<T, true>(out, cond, a, b, ndims);
    } else {
        select_launcher<T, false>(out, cond, a, b, ndims);
    }
}

template <typename T, bool flip>
void select_scalar(Param out, Param cond, Param a, const double b, int ndims) {
    std::string refName = std::string("select_scalar_kernel_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(flip);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D flip=" << flip
                << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {select_cl};
        const int ker_lens[]   = {select_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "select_scalar_kernel");

        addKernelToCache(device, refName, entry);
    }

    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    NDRange local(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    NDRange global(groups_0 * out.info.dims[2] * local[0],
                   groups_1 * out.info.dims[3] * local[1]);

    auto selectOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                  KParam, T, int, int>(*entry.ker);

    selectOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *cond.data, cond.info, *a.data, a.info, scalar<T>(b), groups_0,
             groups_1);
}
}  // namespace kernel
}  // namespace opencl
