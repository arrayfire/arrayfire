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
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/laswp.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr int MAX_PIVOTS = 32;

typedef struct {
    int npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;

template<typename T>
void laswp(int n, cl_mem in, size_t offset, int ldda, int k1, int k2,
           const int *ipiv, int inci, cl::CommandQueue &queue) {
    constexpr int NTHREADS = 256;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(MAX_PIVOTS),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto laswpOp = common::getKernel("laswp", {{laswp_cl_src}}, targs, options);

    int groups = divup(n, NTHREADS);
    cl::NDRange local(NTHREADS);
    cl::NDRange global(groups * local[0]);
    zlaswp_params_t params;

    // retain the cl_mem object during cl::Buffer creation
    cl::Buffer inObj(in, true);

    for (int k = k1 - 1; k < k2; k += MAX_PIVOTS) {
        int pivots_left = k2 - k;

        params.npivots = pivots_left > MAX_PIVOTS ? MAX_PIVOTS : pivots_left;

        for (int j = 0; j < params.npivots; ++j)
            params.ipiv[j] = ipiv[(k + j) * inci] - k - 1;

        unsigned long long k_offset = offset + k * ldda;

        laswpOp(cl::EnqueueArgs(queue, global, local), n, inObj, k_offset, ldda,
                params);
    }
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
