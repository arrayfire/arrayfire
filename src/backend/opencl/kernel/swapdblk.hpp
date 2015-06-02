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
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
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
              cl_mem dB, size_t dB_offset, int lddb, int incb)
{

    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static std::map<int, Program*>  swpProgs;
    static std::map<int, Kernel*> swpKernels;

    int device = getActiveDeviceId();

    std::call_once(compileFlags[device], [device] () {

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName();

            if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            cl::Program prog;
            buildProgram(prog, swapdblk_cl, swapdblk_cl_len, options.str());
            swpProgs[device] = new Program(prog);

            swpKernels[device] = new Kernel(*swpProgs[device], "swapdblk");
        });

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
    auto swapdOp = make_kernel<int,
                               cl_mem, unsigned long long, int, int,
                               cl_mem, unsigned long long, int, int>(*swpKernels[device]);

    swapdOp(EnqueueArgs(getQueue(), global, local),
            nb,
            dA, dA_offset, ldda, inca,
            dB, dB_offset, lddb, incb);

}

}
}
