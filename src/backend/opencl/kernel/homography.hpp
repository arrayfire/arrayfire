/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/ireduce.hpp>
#include <kernel/reduce.hpp>
#include <kernel/sort.hpp>
#include <kernel_headers/homography.hpp>
#include <memory.hpp>
#include <af/defines.h>

#include <limits>
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
constexpr int HG_THREADS_X = 16;
constexpr int HG_THREADS_Y = 16;
constexpr int HG_THREADS   = 256;

template<typename T>
std::array<Kernel, 5> getHomographyKernels(const af_homography_type htype) {
    std::vector<TemplateArg> targs   = {TemplateTypename<T>(),
                                        TemplateArg(htype)};
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());
    options.emplace_back(
        DefineKeyValue(EPS, (std::is_same<T, double>::value
                                 ? std::numeric_limits<double>::epsilon()
                                 : std::numeric_limits<float>::epsilon())));
    if (htype == AF_HOMOGRAPHY_RANSAC) {
        options.emplace_back(DefineKey(RANSAC));
    }
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        options.emplace_back(DefineKey(LMEDS));
    }
    if (getActiveDeviceType() == CL_DEVICE_TYPE_CPU) {
        options.emplace_back(DefineKey(IS_CPU));
    }
    return {
        common::getKernel("compute_homography", {{homography_cl_src}}, targs,
                          options),
        common::getKernel("eval_homography", {{homography_cl_src}}, targs,
                          options),
        common::getKernel("compute_median", {{homography_cl_src}}, targs,
                          options),
        common::getKernel("find_min_median", {{homography_cl_src}}, targs,
                          options),
        common::getKernel("compute_lmeds_inliers", {{homography_cl_src}}, targs,
                          options),
    };
}

template<typename T>
int computeH(Param bestH, Param H, Param err, Param x_src, Param y_src,
             Param x_dst, Param y_dst, Param rnd, const unsigned iterations,
             const unsigned nsamples, const float inlier_thr,
             const af_homography_type htype) {
    using cl::Buffer;
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto kernels = getHomographyKernels<T>(htype);
    auto chOp    = kernels[0];
    auto ehOp    = kernels[1];
    auto cmOp    = kernels[2];
    auto fmOp    = kernels[3];
    auto clOp    = kernels[4];

    const int blk_x_ch = 1;
    const int blk_y_ch = divup(iterations, HG_THREADS_Y);
    const NDRange local_ch(HG_THREADS_X, HG_THREADS_Y);
    const NDRange global_ch(blk_x_ch * HG_THREADS_X, blk_y_ch * HG_THREADS_Y);

    // Build linear system and solve SVD
    chOp(EnqueueArgs(getQueue(), global_ch, local_ch), *H.data, H.info,
         *x_src.data, *y_src.data, *x_dst.data, *y_dst.data, *rnd.data,
         rnd.info, iterations);
    CL_DEBUG_FINISH(getQueue());

    const int blk_x_eh = divup(iterations, HG_THREADS);
    const NDRange local_eh(HG_THREADS);
    const NDRange global_eh(blk_x_eh * HG_THREADS);

    // Allocate some temporary buffers
    Param inliers, idx, median;
    inliers.info.offset = idx.info.offset = median.info.offset = 0;
    inliers.info.dims[0]    = (htype == AF_HOMOGRAPHY_RANSAC)
                                  ? blk_x_eh
                                  : divup(nsamples, HG_THREADS);
    inliers.info.strides[0] = 1;
    idx.info.dims[0] = median.info.dims[0] = blk_x_eh;
    idx.info.strides[0] = median.info.strides[0] = 1;
    for (int k = 1; k < 4; k++) {
        inliers.info.dims[k] = 1;
        inliers.info.strides[k] =
            inliers.info.dims[k - 1] * inliers.info.strides[k - 1];
        idx.info.dims[k] = median.info.dims[k] = 1;
        idx.info.strides[k]                    = median.info.strides[k] =
            idx.info.dims[k - 1] * idx.info.strides[k - 1];
    }
    idx.data =
        bufferAlloc(idx.info.dims[3] * idx.info.strides[3] * sizeof(unsigned));
    inliers.data = bufferAlloc(inliers.info.dims[3] * inliers.info.strides[3] *
                               sizeof(unsigned));
    if (htype == AF_HOMOGRAPHY_LMEDS)
        median.data = bufferAlloc(median.info.dims[3] * median.info.strides[3] *
                                  sizeof(float));
    else
        median.data = bufferAlloc(sizeof(float));

    // Compute (and for RANSAC, evaluate) homographies
    ehOp(EnqueueArgs(getQueue(), global_eh, local_eh), *inliers.data, *idx.data,
         *H.data, H.info, *err.data, err.info, *x_src.data, *y_src.data,
         *x_dst.data, *y_dst.data, *rnd.data, iterations, nsamples, inlier_thr);
    CL_DEBUG_FINISH(getQueue());

    unsigned inliersH, idxH;
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        // TODO: Improve this sorting, if the number of iterations is
        // sufficiently large, this can be *very* slow
        kernel::sort0<float>(err, true);

        unsigned minIdx;
        float minMedian;

        // Compute median of every iteration
        cmOp(EnqueueArgs(getQueue(), global_eh, local_eh), *median.data,
             *idx.data, *err.data, err.info, iterations);
        CL_DEBUG_FINISH(getQueue());

        // Reduce medians, only in case iterations > 256
        if (blk_x_eh > 1) {
            const NDRange local_fm(HG_THREADS);
            const NDRange global_fm(HG_THREADS);

            Buffer* finalMedian = bufferAlloc(sizeof(float));
            Buffer* finalIdx    = bufferAlloc(sizeof(unsigned));

            fmOp(EnqueueArgs(getQueue(), global_fm, local_fm), *finalMedian,
                 *finalIdx, *median.data, median.info, *idx.data);
            CL_DEBUG_FINISH(getQueue());

            getQueue().enqueueReadBuffer(*finalMedian, CL_TRUE, 0,
                                         sizeof(float), &minMedian);
            getQueue().enqueueReadBuffer(*finalIdx, CL_TRUE, 0,
                                         sizeof(unsigned), &minIdx);

            bufferFree(finalMedian);
            bufferFree(finalIdx);
        } else {
            getQueue().enqueueReadBuffer(*median.data, CL_TRUE, 0,
                                         sizeof(float), &minMedian);
            getQueue().enqueueReadBuffer(*idx.data, CL_TRUE, 0,
                                         sizeof(unsigned), &minIdx);
        }

        // Copy best homography to output
        getQueue().enqueueCopyBuffer(*H.data, *bestH.data,
                                     minIdx * 9 * sizeof(T), 0, 9 * sizeof(T));

        const int blk_x_cl = divup(nsamples, HG_THREADS);
        const NDRange local_cl(HG_THREADS);
        const NDRange global_cl(blk_x_cl * HG_THREADS);

        clOp(EnqueueArgs(getQueue(), global_cl, local_cl), *inliers.data,
             *bestH.data, *x_src.data, *y_src.data, *x_dst.data, *y_dst.data,
             minMedian, nsamples);
        CL_DEBUG_FINISH(getQueue());

        // Adds up the total number of inliers
        Param totalInliers;
        totalInliers.info.offset = 0;
        for (int k = 0; k < 4; k++)
            totalInliers.info.dims[k] = totalInliers.info.strides[k] = 1;
        totalInliers.data = bufferAlloc(sizeof(unsigned));

        kernel::reduce<unsigned, unsigned, af_add_t>(totalInliers, inliers, 0,
                                                     false, 0.0);

        getQueue().enqueueReadBuffer(*totalInliers.data, CL_TRUE, 0,
                                     sizeof(unsigned), &inliersH);

        bufferFree(totalInliers.data);
    } else /* if (htype == AF_HOMOGRAPHY_RANSAC) */ {
        unsigned blockIdx;
        inliersH = kernel::ireduceAll<unsigned, af_max_t>(&blockIdx, inliers);

        // Copies back index and number of inliers of best homography estimation
        getQueue().enqueueReadBuffer(*idx.data, CL_TRUE,
                                     blockIdx * sizeof(unsigned),
                                     sizeof(unsigned), &idxH);
        getQueue().enqueueCopyBuffer(*H.data, *bestH.data, idxH * 9 * sizeof(T),
                                     0, 9 * sizeof(T));
    }

    bufferFree(inliers.data);
    bufferFree(idx.data);
    bufferFree(median.data);

    return (int)inliersH;
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
