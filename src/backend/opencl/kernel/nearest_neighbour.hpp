/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <program.hpp>
#include <common/dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/nearest_neighbour.hpp>
#include <memory.hpp>
#include <math.hpp>
#include <common/dispatch.hpp>
#include <cache.hpp>

using cl::KernelFunctor;
using cl::LocalSpaceArg;

namespace opencl
{

namespace kernel
{

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type>
void nearest_neighbour(Param idx,
                       Param dist,
                       Param query,
                       Param train,
                       const dim_t dist_dim,
                       const unsigned n_dist)
{
    const unsigned feat_len = query.info.dims[dist_dim];
    const To max_dist = maxval<To>();

    // Determine maximum feat_len capable of using shared memory (faster)
    cl_ulong avail_lmem = getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    size_t lmem_predef = 2 * THREADS * sizeof(unsigned) + feat_len * sizeof(T);
    size_t ltrain_sz = THREADS * feat_len * sizeof(T);
    bool use_lmem = (avail_lmem >= (lmem_predef + ltrain_sz)) ? true : false;
    size_t lmem_sz = (use_lmem) ? lmem_predef + ltrain_sz : lmem_predef;

    unsigned unroll_len = nextpow2(feat_len);
    if (unroll_len != feat_len) unroll_len = 0;

    std::string ref_name =
        std::string("knn_") +
        std::to_string(dist_type) +
        std::string("_") +
        std::to_string(use_lmem) +
        std::string("_") +
        std::string(dtype_traits<T>::getName()) +
        std::string("_") +
        std::to_string(unroll_len);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog==0 && entry.ker==0) {

        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
            << " -D To=" << dtype_traits<To>::getName()
            << " -D THREADS=" << THREADS
            << " -D FEAT_LEN=" << unroll_len;

        switch(dist_type) {
            case AF_SAD: options <<" -D DISTOP=_sad_"; break;
            case AF_SSD: options <<" -D DISTOP=_ssd_"; break;
            case AF_SHD: options <<" -D DISTOP=_shd_ -D __SHD__";
                         break;
            default: break;
        }

        if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        if (use_lmem)
            options << " -D USE_LOCAL_MEM";

        cl::Program prog;
        buildProgram(prog,
                nearest_neighbour_cl,
                nearest_neighbour_cl_len,
                options.str());

        entry.prog = new Program(prog);
        entry.ker = new Kernel[3];

        entry.ker[0] = Kernel(*entry.prog, "nearest_neighbour_unroll");
        entry.ker[1] = Kernel(*entry.prog, "nearest_neighbour");
        entry.ker[2] = Kernel(*entry.prog, "select_matches");

        addKernelToCache(device, ref_name, entry);
    }

    const dim_t sample_dim = (dist_dim == 0) ? 1 : 0;

    const unsigned nquery = query.info.dims[sample_dim];
    const unsigned ntrain = train.info.dims[sample_dim];

    unsigned nblk = divup(ntrain, THREADS);
    const NDRange local(THREADS, 1);
    const NDRange global(nblk * THREADS, 1);

    cl::Buffer *d_blk_idx  = bufferAlloc(nblk * nquery * sizeof(unsigned));
    cl::Buffer *d_blk_dist = bufferAlloc(nblk * nquery * sizeof(To));

    // For each query vector, find training vector with smallest Hamming
    // distance per CUDA block
    if (unroll_len > 0) {
        auto huOp = KernelFunctor<Buffer, Buffer,
                                Buffer, KParam,
                                Buffer, KParam,
                                const To,
                                LocalSpaceArg> (entry.ker[0]);

        huOp(EnqueueArgs(getQueue(), global, local),
              *d_blk_idx, *d_blk_dist,
              *query.data, query.info, *train.data, train.info,
              max_dist, cl::Local(lmem_sz));
    }
    else {
        auto hmOp = KernelFunctor<Buffer, Buffer,
                                Buffer, KParam,
                                Buffer, KParam,
                                const To, const unsigned,
                                LocalSpaceArg> (entry.ker[1]);

        hmOp(EnqueueArgs(getQueue(), global, local),
              *d_blk_idx, *d_blk_dist,
              *query.data, query.info, *train.data, train.info,
              max_dist, feat_len, cl::Local(lmem_sz));
    }
    CL_DEBUG_FINISH(getQueue());

    const NDRange local_sm(32, 8);
    const NDRange global_sm(divup(nquery, 32) * 32, 8);

    // Reduce all smallest Hamming distances from each block and store final
    // best match
    auto smOp = KernelFunctor<Buffer, Buffer, Buffer, Buffer,
                            const unsigned, const unsigned,
                            const To> (entry.ker[2]);

    smOp(EnqueueArgs(getQueue(), global_sm, local_sm),
          *idx.data, *dist.data,
          *d_blk_idx, *d_blk_dist,
          nquery, nblk, max_dist);
    CL_DEBUG_FINISH(getQueue());

    bufferFree(d_blk_idx);
    bufferFree(d_blk_dist);
}

} // namespace kernel

} // namespace opencl
