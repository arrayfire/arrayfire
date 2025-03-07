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
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/swapdblk.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void swapdblk(int n, int nb, cl_mem dA, size_t dA_offset, int ldda, int inca,
              cl_mem dB, size_t dB_offset, int lddb, int incb,
              cl_command_queue queue) {
    using cl::Buffer;
    using cl::CommandQueue;
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    int nblocks = n / nb;
    if (nblocks == 0) return;

    vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto swapdblk =
        common::getKernel("swapdblk", {{swapdblk_cl_src}}, targs, compileOpts);

    int info = 0;
    if (n < 0) {
        info = -1;
    } else if (nb < 1 || nb > 1024) {
        info = -2;
    } else if (ldda < (nblocks - 1) * nb * inca + nb) {
        info = -4;
    } else if (inca < 0) {
        info = -5;
    } else if (lddb < (nblocks - 1) * nb * incb + nb) {
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

    Buffer dAObj(dA, true);
    Buffer dBObj(dB, true);

    CommandQueue q(queue, true);
    swapdblk(EnqueueArgs(q, global, local), nb, dAObj, dA_offset, ldda, inca,
             dBObj, dB_offset, lddb, incb);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
