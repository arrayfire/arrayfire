/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/laset_band.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>
#include <traits.hpp>

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
#if 0 // Needs to be enabled when unmqr2 is enabled
static const int NB = 64;
template<int num>
const char *laset_band_name() { return "laset_none"; }
template<> const char *laset_band_name<0>() { return "laset_band_lower"; }
template<> const char *laset_band_name<1>() { return "laset_band_upper"; }

template<typename T, int uplo>
void laset_band(int m, int  n, int k,
                T offdiag, T diag,
                cl_mem dA, size_t dA_offset, magma_int_t ldda)
{
    std::string refName = laset_band_name<uplo>() + std::string("_") +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(uplo);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
            << " -D NB=" << NB
            << " -D IS_CPLX=" << af::iscplx<T>();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {laset_band_cl};
        const int   ker_lens[] = {laset_band_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, laset_band_name<uplo>());

        addKernelToCache(device, refName, entry);
    }

    int threads = 1;
    int groups = 1;

    if (uplo == 0) {
        threads = std::min(k, m);
        groups = (std::min(m, n) - 1) / NB + 1;
    } else {
        threads = std::min(k, n);
        groups = (std::min(m+k-1, n) - 1) / NB + 1;
    }

    NDRange local(threads, 1);
    NDRange global(threads * groups, 1);

    auto lasetBandOp = KernelFunctor<int, int, T, T, cl_mem, unsigned long long, int>(*entry.ker);

    lasetBandOp(EnqueueArgs(getQueue(), global, local), m, n, offdiag, diag, dA, dA_offset, ldda);
}
#endif
}
}
