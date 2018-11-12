/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/swapdblk.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
template<typename T>
void swapdblk(int n, int nb,
              cl_mem dA, size_t dA_offset, int ldda, int inca,
              cl_mem dB, size_t dB_offset, int lddb, int incb,
              cl_command_queue queue)
{
    std::string refName = std::string("swapdblk_") + std::string(dtype_traits<T>::getName());

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {swapdblk_cl};
        const int   ker_lens[] = {swapdblk_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "swapdblk");

        addKernelToCache(device, refName, entry);
    }

    int nblocks = n / nb;

    if(nblocks == 0)
        return;

    int info = 0;
    if (n < 0) {
        info = -1;
    } else if (nb < 1 || nb > 1024) {
        info = -2;
    } else if (ldda < (nblocks-1)*nb*inca + nb) {
        info = -4;
    } else if (inca < 0) {
        info = -5;
    } else if (lddb < (nblocks-1)*nb*incb + nb) {
        info = -7;
    } else if (incb < 0) {
        info = -8;
    }

    if (info != 0) {
        AF_ERROR("Invalid configuration", AF_ERR_INTERNAL);
        return;
    }

    NDRange local(nb);
    NDRange global(nblocks * nb);

    cl::Buffer dAObj(dA, true);
    cl::Buffer dBObj(dB, true);

    auto swapdOp = KernelFunctor<int, Buffer, unsigned long long, int, int,
                               Buffer, unsigned long long, int, int>(*entry.ker);

    cl::CommandQueue q(queue);
    swapdOp(EnqueueArgs(q, global, local),
            nb, dAObj, dA_offset, ldda, inca, dBObj, dB_offset, lddb, incb);
}
}
}
