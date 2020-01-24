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
#include <kernel_headers/laswp.hpp>
#include <platform.hpp>
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
static const int NTHREADS   = 256;
static const int MAX_PIVOTS = 32;

typedef struct {
    int npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;

template<typename T>
void laswp(int n, cl_mem in, size_t offset, int ldda, int k1, int k2,
           const int *ipiv, int inci, cl::CommandQueue &queue) {
    std::string refName =
        std::string("laswp_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D MAX_PIVOTS=" << MAX_PIVOTS;

        options << getTypeBuildDefinition<T>();

        const char *ker_strs[] = {laswp_cl};
        const int ker_lens[]   = {laswp_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "laswp");

        addKernelToCache(device, refName, entry);
    }

    int groups = divup(n, NTHREADS);
    NDRange local(NTHREADS);
    NDRange global(groups * local[0]);
    zlaswp_params_t params;

    // retain the cl_mem object during cl::Buffer creation
    cl::Buffer inObj(in, true);

    auto laswpOp =
        KernelFunctor<int, Buffer, unsigned long long, int, zlaswp_params_t>(
            *entry.ker);

    for (int k = k1 - 1; k < k2; k += MAX_PIVOTS) {
        int pivots_left = k2 - k;

        params.npivots = pivots_left > MAX_PIVOTS ? MAX_PIVOTS : pivots_left;

        for (int j = 0; j < params.npivots; ++j)
            params.ipiv[j] = ipiv[(k + j) * inci] - k - 1;

        unsigned long long k_offset = offset + k * ldda;

        laswpOp(EnqueueArgs(queue, global, local), n, inObj, k_offset, ldda,
                params);
    }
}
}  // namespace kernel
}  // namespace opencl
