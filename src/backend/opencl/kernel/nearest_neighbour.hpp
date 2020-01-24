/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/nearest_neighbour.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <af/defines.h>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::LocalSpaceArg;
using cl::NDRange;
using cl::Program;

namespace opencl {

namespace kernel {

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type>
void all_distances(Param dist, Param query, Param train, const dim_t dist_dim) {
    const dim_t feat_len = query.info.dims[dist_dim];
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

    std::string ref_name = std::string("knn_") + std::to_string(dist_type) +
                           std::string("_") + std::to_string(use_lmem) +
                           std::string("_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string("_") + std::to_string(unroll_len);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D To=" << dtype_traits<To>::getName()
                << " -D THREADS=" << THREADS << " -D FEAT_LEN=" << unroll_len;

        switch (dist_type) {
            case AF_SAD: options << " -D DISTOP=_sad_"; break;
            case AF_SSD: options << " -D DISTOP=_ssd_"; break;
            case AF_SHD: options << " -D DISTOP=_shd_ -D __SHD__"; break;
            default: break;
        }

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        if (use_lmem) options << " -D USE_LOCAL_MEM";

        cl::Program prog;
        buildProgram(prog, nearest_neighbour_cl, nearest_neighbour_cl_len,
                     options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel;

        *entry.ker = Kernel(*entry.prog, "all_distances");

        addKernelToCache(device, ref_name, entry);
    }

    const dim_t sample_dim = (dist_dim == 0) ? 1 : 0;

    const unsigned ntrain = train.info.dims[sample_dim];

    unsigned nblk = divup(ntrain, THREADS);
    const NDRange local(THREADS, 1);
    const NDRange global(nblk * THREADS, 1);

    // For each query vector, find training vector with smallest Hamming
    // distance per CUDA block
    auto hmOp = KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, const To,
                              const unsigned, const unsigned, const unsigned,
                              LocalSpaceArg>(*entry.ker);

    for (dim_t feat_offset = 0; feat_offset < feat_len;
         feat_offset += THREADS) {
        hmOp(EnqueueArgs(getQueue(), global, local), *dist.data, *query.data,
             query.info, *train.data, train.info, max_dist, feat_len,
             max_kern_feat_len, feat_offset, cl::Local(lmem_sz));
        CL_DEBUG_FINISH(getQueue());
    }
}

}  // namespace kernel

}  // namespace opencl
