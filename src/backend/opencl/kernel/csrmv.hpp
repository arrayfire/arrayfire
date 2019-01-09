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
#include <kernel_headers/csrmv.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <af/opencl.h>
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
static const int MAX_CSRMV_GROUPS = 4096;
template <typename T>
void csrmv(Param out, const Param &values, const Param &rowIdx,
           const Param &colIdx, const Param &rhs, const T alpha, const T beta) {
    bool use_alpha = (alpha != scalar<T>(1.0));
    bool use_beta  = (beta != scalar<T>(0.0));

    // Using greedy indexing is causing performance issues on many platforms
    // FIXME: Figure out why
    bool use_greedy = false;

    // FIXME: Find a better number based on average non zeros per row
    int threads = 64;

    std::string ref_name =
        std::string("csrmv_") + std::string(dtype_traits<T>::getName()) +
        std::string("_") + std::to_string(use_alpha) + std::string("_") +
        std::to_string(use_beta) + std::string("_") +
        std::to_string(use_greedy) + std::string("_") + std::to_string(threads);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << " -D USE_ALPHA=" << use_alpha;
        options << " -D USE_BETA=" << use_beta;
        options << " -D USE_GREEDY=" << use_greedy;
        options << " -D THREADS=" << threads;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }
        if (std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value) {
            options << " -D IS_CPLX=1";
        } else {
            options << " -D IS_CPLX=0";
        }

        const char *ker_strs[] = {csrmv_cl};
        const int ker_lens[]   = {csrmv_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog   = new Program(prog);
        entry.ker    = new Kernel[2];
        entry.ker[0] = Kernel(*entry.prog, "csrmv_thread");
        entry.ker[1] = Kernel(*entry.prog, "csrmv_block");

        addKernelToCache(device, ref_name, entry);
    }

    int count           = 0;
    cl::Buffer *counter = bufferAlloc(sizeof(int));
    getQueue().enqueueWriteBuffer(*counter, CL_TRUE, 0, sizeof(int),
                                  (void *)&count);

    // TODO: Figure out the proper way to choose either csrmv_thread or
    // csrmv_block
    bool is_csrmv_block = true;
    auto csrmv_kernel   = is_csrmv_block ? entry.ker[1] : entry.ker[0];
    auto csrmv_func = KernelFunctor<Buffer, Buffer, Buffer, Buffer, int, Buffer,
                                    KParam, T, T, Buffer>(csrmv_kernel);

    NDRange local(is_csrmv_block ? threads : THREADS_PER_GROUP, 1);
    int M = rowIdx.info.dims[0] - 1;

    int groups_x =
        is_csrmv_block ? divup(M, REPEAT) : divup(M, REPEAT * local[0]);
    groups_x = std::min(groups_x, MAX_CSRMV_GROUPS);
    NDRange global(local[0] * groups_x, 1);

    csrmv_func(EnqueueArgs(getQueue(), global, local), *out.data, *values.data,
               *rowIdx.data, *colIdx.data, M, *rhs.data, rhs.info, alpha, beta,
               *counter);

    CL_DEBUG_FINISH(getQueue());
    bufferFree(counter);
}
}  // namespace kernel
}  // namespace opencl
