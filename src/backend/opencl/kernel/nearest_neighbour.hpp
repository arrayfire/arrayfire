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
#include <kernel_headers/nearest_neighbour.hpp>
#include <math.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, typename To>
void allDistances(Param dist, Param query, Param train, const dim_t dist_dim,
                  af_match_type dist_type) {
    constexpr unsigned THREADS = 256;

    const unsigned feat_len = static_cast<uint>(query.info.dims[dist_dim]);
    const unsigned max_kern_feat_len =
        min(THREADS, static_cast<unsigned>(feat_len));
    const To max_dist = maxval<To>();

    // Determine maximum feat_len capable of using shared memory (faster)
    cl_ulong avail_lmem = getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    size_t lmem_predef =
        2 * THREADS * sizeof(unsigned) + max_kern_feat_len * sizeof(T);
    size_t ltrain_sz = THREADS * max_kern_feat_len * sizeof(T);
    bool use_lmem    = (avail_lmem >= (lmem_predef + ltrain_sz)) ? true : false;
    size_t lmem_sz   = (use_lmem) ? lmem_predef + ltrain_sz : lmem_predef;

    unsigned unroll_len = nextpow2(feat_len);
    if (unroll_len != feat_len) unroll_len = 0;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(dist_type),
        TemplateArg(use_lmem),
        TemplateArg(unroll_len),
    };

    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineValue(THREADS),
        DefineKeyValue(FEAT_LEN, unroll_len),
    };
    options.emplace_back(getTypeBuildDefinition<T>());
    if (use_lmem) { options.emplace_back(DefineKey(USE_LOCAL_MEM)); }
    if (dist_type == AF_SAD) {
        options.emplace_back(DefineKeyValue(DISTOP, "_sad_"));
    }
    if (dist_type == AF_SSD) {
        options.emplace_back(DefineKeyValue(DISTOP, "_ssd_"));
    }
    if (dist_type == AF_SHD) {
        options.emplace_back(DefineKeyValue(DISTOP, "_shd_"));
        options.emplace_back(DefineKey(__SHD__));
    }
    auto hmOp = common::getKernel("knnAllDistances",
                                  {{nearest_neighbour_cl_src}}, targs, options);

    const dim_t sample_dim = (dist_dim == 0) ? 1 : 0;

    const unsigned ntrain = train.info.dims[sample_dim];

    unsigned nblk = divup(ntrain, THREADS);
    const cl::NDRange local(THREADS, 1);
    const cl::NDRange global(nblk * THREADS, 1);

    // For each query vector, find training vector with smallest Hamming
    // distance per CUDA block
    for (uint feat_offset = 0; feat_offset < feat_len; feat_offset += THREADS) {
        hmOp(cl::EnqueueArgs(getQueue(), global, local), *dist.data,
             *query.data, query.info, *train.data, train.info, max_dist,
             feat_len, max_kern_feat_len, feat_offset, cl::Local(lmem_sz));
        CL_DEBUG_FINISH(getQueue());
    }
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
