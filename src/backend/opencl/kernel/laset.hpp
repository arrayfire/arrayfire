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
#include <kernel_headers/laset.hpp>
#include <magma_types.h>
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
static const int BLK_X = 64;
static const int BLK_Y = 32;

template<int num>
const char *laset_name() {
    return "laset_none";
}
template<>
const char *laset_name<0>() {
    return "laset_full";
}
template<>
const char *laset_name<1>() {
    return "laset_lower";
}
template<>
const char *laset_name<2>() {
    return "laset_upper";
}

template<typename T, int uplo>
void laset(int m, int n, T offdiag, T diag, cl_mem dA, size_t dA_offset,
           magma_int_t ldda, cl_command_queue queue) {
    std::string refName = laset_name<uplo>() + std::string("_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(uplo);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D BLK_X=" << BLK_X << " -D BLK_Y=" << BLK_Y
                << " -D IS_CPLX=" << af::iscplx<T>();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char *ker_strs[] = {laset_cl};
        const int ker_lens[]   = {laset_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, laset_name<uplo>());

        addKernelToCache(device, refName, entry);
    }

    int groups_x = (m - 1) / BLK_X + 1;
    int groups_y = (n - 1) / BLK_Y + 1;

    NDRange local(BLK_X, 1);
    NDRange global(groups_x * local[0], groups_y * local[1]);

    // retain the cl_mem object during cl::Buffer creation
    cl::Buffer dAObj(dA, true);

    auto lasetOp =
        KernelFunctor<int, int, T, T, Buffer, unsigned long long, int>(
            *entry.ker);

    cl::CommandQueue q(queue);
    lasetOp(EnqueueArgs(q, global, local), m, n, offdiag, diag, dAObj,
            dA_offset, ldda);
}
}  // namespace kernel
}  // namespace opencl
