/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/laset.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>
#include <traits.hpp>

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

static const int BLK_X = 64;
static const int BLK_Y = 32;

template<int num>
const char *laset_name() { return "laset_none"; }
template<> const char *laset_name<0>() { return "laset_full"; }
template<> const char *laset_name<1>() { return "laset_lower"; }
template<> const char *laset_name<2>() { return "laset_upper"; }

template<typename T, int uplo>
void laset(int m, int  n,
           T offdiag, T diag,
           cl_mem dA, size_t dA_offset, magma_int_t ldda)
{

    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static std::map<int, Program*>  setProgs;
    static std::map<int, Kernel*> setKernels;

    int device = getActiveDeviceId();

    std::call_once(compileFlags[device], [device] () {

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName()
                    << " -D BLK_X=" << BLK_X
                    << " -D BLK_Y=" << BLK_Y
                    << " -D IS_CPLX=" << af::iscplx<T>();

            if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            cl::Program prog;
            buildProgram(prog, laset_cl, laset_cl_len, options.str());
            setProgs[device] = new Program(prog);
            setKernels[device] = new Kernel(*setProgs[device], laset_name<uplo>());
        });

    int groups_x = (m - 1) / BLK_X + 1;
    int groups_y = (n - 1) / BLK_Y + 1;

    NDRange local(BLK_X, 1);
    NDRange global(groups_x * local[0],
                   groups_y * local[1]);

    auto lasetOp = make_kernel<int, int, T, T, cl_mem, unsigned long long, int>(*setKernels[device]);
    lasetOp(EnqueueArgs(getQueue(), global, local),
            m, n, offdiag, diag, dA, dA_offset, ldda);
}

}
}
