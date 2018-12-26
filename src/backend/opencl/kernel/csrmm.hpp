/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#pragma once
#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/csrmm.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <map>
#include <mutex>
#include <string>
#include "config.hpp"
#include "reduce.hpp"
#include "scan_dim.hpp"
#include "scan_first.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int MAX_CSRMM_GROUPS = 4096;
template<typename T>
void csrmm_nt(Param out, const Param &values, const Param &rowIdx,
              const Param &colIdx, const Param &rhs, const T alpha,
              const T beta) {
    bool use_alpha = (alpha != scalar<T>(1.0));
    bool use_beta  = (beta != scalar<T>(0.0));

    // Using greedy indexing is causing performance issues on many platforms
    // FIXME: Figure out why
    bool use_greedy = false;

    std::string ref_name = std::string("csrmm_nt_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string("_") + std::to_string(use_alpha) +
                           std::string("_") + std::to_string(use_beta) +
                           std::string("_") + std::to_string(use_greedy);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << " -D USE_ALPHA=" << use_alpha;
        options << " -D USE_BETA=" << use_beta;
        options << " -D USE_GREEDY=" << use_greedy;
        options << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }
        if (std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value) {
            options << " -D IS_CPLX=1";
        } else {
            options << " -D IS_CPLX=0";
        }

        const char *ker_strs[] = {csrmm_cl};
        const int ker_lens[]   = {csrmm_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog   = new Program(prog);
        entry.ker    = new Kernel[2];
        entry.ker[0] = Kernel(*entry.prog, "csrmm_nt");
        // FIXME: Change this after adding another kernel
        entry.ker[1] = Kernel(*entry.prog, "csrmm_nt");

        addKernelToCache(device, ref_name, entry);
    }

    auto csrmm_nt_kernel = entry.ker[0];
    auto csrmm_nt_func =
        KernelFunctor<Buffer, Buffer, Buffer, Buffer, int, int, Buffer, KParam,
                      T, T, Buffer>(csrmm_nt_kernel);
    NDRange local(THREADS_PER_GROUP, 1);
    int M = rowIdx.info.dims[0] - 1;
    int N = rhs.info.dims[0];

    int groups_x = divup(N, local[0]);
    int groups_y = divup(M, REPEAT);
    groups_y     = std::min(groups_y, MAX_CSRMM_GROUPS);
    NDRange global(local[0] * groups_x, local[1] * groups_y);

    std::vector<int> count(groups_x);
    cl::Buffer *counter = bufferAlloc(count.size() * sizeof(int));
    getQueue().enqueueWriteBuffer(
        *counter, CL_TRUE, 0, count.size() * sizeof(int), (void *)count.data());

    csrmm_nt_func(EnqueueArgs(getQueue(), global, local), *out.data,
                  *values.data, *rowIdx.data, *colIdx.data, M, N, *rhs.data,
                  rhs.info, alpha, beta, *counter);

    bufferFree(counter);
}
}  // namespace kernel
}  // namespace opencl
