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
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/nearest_neighbour.hpp>
#include <memory.hpp>
#include <math.hpp>

using cl::LocalSpaceArg;

namespace opencl
{

namespace kernel
{

static const unsigned THREADS = 256;

template<typename T, typename To, af_match_type dist_type, bool use_lmem, unsigned unroll_len>
void nearest_neighbour(Param idx,
                       Param dist,
                       Param query,
                       Param train,
                       const dim_t dist_dim,
                       const unsigned n_dist,
                       const size_t lmem_sz)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program        nearest_neighbourProgs[DeviceManager::MAX_DEVICES];
        static Kernel             huKernel[DeviceManager::MAX_DEVICES];
        static Kernel             hmKernel[DeviceManager::MAX_DEVICES];
        static Kernel             smKernel[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        const unsigned feat_len = query.info.dims[dist_dim];
        const To max_dist = limit_max<To>();

        if (feat_len > THREADS) {
            OPENCL_NOT_SUPPORTED();
        }

        std::call_once( compileFlags[device], [device] () {

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

                buildProgram(nearest_neighbourProgs[device],
                             nearest_neighbour_cl,
                             nearest_neighbour_cl_len,
                             options.str());

                huKernel[device] = Kernel(nearest_neighbourProgs[device], "nearest_neighbour_unroll");
                hmKernel[device] = Kernel(nearest_neighbourProgs[device], "nearest_neighbour");
                smKernel[device] = Kernel(nearest_neighbourProgs[device], "select_matches");
            });

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
            auto huOp = make_kernel<Buffer, Buffer,
                                    Buffer, KParam,
                                    Buffer, KParam,
                                    const To,
                                    LocalSpaceArg> (huKernel[device]);

            huOp(EnqueueArgs(getQueue(), global, local),
                 *d_blk_idx, *d_blk_dist,
                 *query.data, query.info, *train.data, train.info,
                 max_dist, cl::Local(lmem_sz));
        }
        else {
            auto hmOp = make_kernel<Buffer, Buffer,
                                    Buffer, KParam,
                                    Buffer, KParam,
                                    const To, const unsigned,
                                    LocalSpaceArg> (hmKernel[device]);

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
        auto smOp = make_kernel<Buffer, Buffer, Buffer, Buffer,
                                const unsigned, const unsigned,
                                const To> (smKernel[device]);

        smOp(EnqueueArgs(getQueue(), global_sm, local_sm),
             *idx.data, *dist.data,
             *d_blk_idx, *d_blk_dist,
             nquery, nblk, max_dist);
        CL_DEBUG_FINISH(getQueue());

        bufferFree(d_blk_idx);
        bufferFree(d_blk_dist);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

} // namespace kernel

} // namespace opencl
