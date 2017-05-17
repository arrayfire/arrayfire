/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/lu_split.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>
#include <math.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using af::scalar_to_option;

namespace opencl
{
namespace kernel
{
// Kernel Launch Config Values
static const unsigned TX = 32;
static const unsigned TY = 8;
static const unsigned TILEX = 128;
static const unsigned TILEY = 32;

template<typename T, bool same_dims>
void lu_split_launcher(Param lower, Param upper, const Param in)
{
    std::string refName = std::string("lu_split_kernel_") +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(same_dims);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
            << " -D same_dims=" << same_dims
            << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")"
            << " -D ONE=(T)(" << scalar_to_option(scalar<T>(1)) << ")";

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {lu_split_cl};
        const int   ker_lens[] = {lu_split_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "lu_split_kernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(TX, TY);

    int groups_x = divup(in.info.dims[0], TILEX);
    int groups_y = divup(in.info.dims[1], TILEY);

    NDRange global(groups_x * local[0] * in.info.dims[2], groups_y * local[1] * in.info.dims[3]);

    auto lu_split_op = KernelFunctor<Buffer, const KParam, Buffer, const KParam,
                                    const Buffer, const KParam, const int, const int> (*entry.ker);

    lu_split_op(EnqueueArgs(getQueue(), global, local),
                *lower.data, lower.info, *upper.data, upper.info,
                *in.data, in.info, groups_x, groups_y);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void lu_split(Param lower, Param upper, const Param in)
{
    bool same_dims =
        (lower.info.dims[0] == in.info.dims[0]) &&
        (lower.info.dims[1] == in.info.dims[1]);

    if (same_dims) {
        lu_split_launcher<T, true >(lower, upper, in);
    } else {
        lu_split_launcher<T, false>(lower, upper, in);
    }
}
}
}
