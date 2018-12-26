/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <kernel/ireduce.hpp>
#include <kernel/reduce.hpp>
#include <kernel/sort.hpp>
#include <kernel_headers/homography.hpp>
#include <memory.hpp>
#include <af/defines.h>
#include <cfloat>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::LocalSpaceArg;
using cl::NDRange;
using cl::Program;
using std::vector;

namespace opencl {
namespace kernel {
const int HG_THREADS_X = 16;
const int HG_THREADS_Y = 16;
const int HG_THREADS   = 256;

template<typename T, af_homography_type htype>
std::array<cl::Kernel*, 5> getHomographyKernels() {
    static const unsigned NUM_KERNELS           = 5;
    static const char* kernelNames[NUM_KERNELS] = {
        "compute_homography", "eval_homography", "compute_median",
        "find_min_median", "compute_lmeds_inliers"};

    kc_entry_t entries[NUM_KERNELS];

    int device = getActiveDeviceId();

    std::string checkName = kernelNames[0] + std::string("_") +
                            std::string(dtype_traits<T>::getName()) +
                            std::to_string(htype);

    entries[0] = kernelCache(device, checkName);

    if (entries[0].prog == 0 && entries[0].ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();

        if (std::is_same<T, double>::value) {
            options << " -D USE_DOUBLE";
            options << " -D EPS=" << DBL_EPSILON;
        } else
            options << " -D EPS=" << FLT_EPSILON;

        if (htype == AF_HOMOGRAPHY_RANSAC)
            options << " -D RANSAC";
        else if (htype == AF_HOMOGRAPHY_LMEDS)
            options << " -D LMEDS";

        if (getActiveDeviceType() == CL_DEVICE_TYPE_CPU) {
            options << " -D IS_CPU";
        }

        cl::Program prog;
        buildProgram(prog, homography_cl, homography_cl_len, options.str());

        for (unsigned i = 0; i < NUM_KERNELS; ++i) {
            entries[i].prog = new Program(prog);
            entries[i].ker  = new Kernel(*entries[i].prog, kernelNames[i]);

            std::string name = kernelNames[i] + std::string("_") +
                               std::string(dtype_traits<T>::getName()) +
                               std::to_string(htype);

            addKernelToCache(device, name, entries[i]);
        }
    } else {
        for (unsigned i = 1; i < NUM_KERNELS; ++i) {
            std::string name = kernelNames[i] + std::string("_") +
                               std::string(dtype_traits<T>::getName()) +
                               std::to_string(htype);

            entries[i] = kernelCache(device, name);
        }
    }

    std::array<cl::Kernel*, NUM_KERNELS> retVal;
    for (unsigned i = 0; i < NUM_KERNELS; ++i) retVal[i] = entries[i].ker;

    return retVal;
}

template<typename T, af_homography_type htype>
int computeH(Param bestH, Param H, Param err, Param x_src, Param y_src,
             Param x_dst, Param y_dst, Param rnd, const unsigned iterations,
             const unsigned nsamples, const float inlier_thr) {
    auto kernels = getHomographyKernels<T, htype>();

    const int blk_x_ch = 1;
    const int blk_y_ch = divup(iterations, HG_THREADS_Y);
    const NDRange local_ch(HG_THREADS_X, HG_THREADS_Y);
    const NDRange global_ch(blk_x_ch * HG_THREADS_X, blk_y_ch * HG_THREADS_Y);

    // Build linear system and solve SVD
    auto chOp = KernelFunctor<Buffer, KParam, Buffer, Buffer, Buffer, Buffer,
                              Buffer, KParam, unsigned>(*kernels[0]);

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
    inliers.info.dims[0] = (htype == AF_HOMOGRAPHY_RANSAC)
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
    auto ehOp = KernelFunctor<Buffer, Buffer, Buffer, KParam, Buffer, KParam,
                              Buffer, Buffer, Buffer, Buffer, Buffer, unsigned,
                              unsigned, float>(*kernels[1]);

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
        auto cmOp = KernelFunctor<Buffer, Buffer, Buffer, KParam, unsigned>(
            *kernels[2]);

        cmOp(EnqueueArgs(getQueue(), global_eh, local_eh), *median.data,
             *idx.data, *err.data, err.info, iterations);

        CL_DEBUG_FINISH(getQueue());

        // Reduce medians, only in case iterations > 256
        if (blk_x_eh > 1) {
            const NDRange local_fm(HG_THREADS);
            const NDRange global_fm(HG_THREADS);

            cl::Buffer* finalMedian = bufferAlloc(sizeof(float));
            cl::Buffer* finalIdx    = bufferAlloc(sizeof(unsigned));

            auto fmOp = KernelFunctor<Buffer, Buffer, Buffer, KParam, Buffer>(
                *kernels[3]);

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

        auto clOp = KernelFunctor<Buffer, Buffer, Buffer, Buffer, Buffer,
                                  Buffer, float, unsigned>(*kernels[4]);

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
    } else if (htype == AF_HOMOGRAPHY_RANSAC) {
        unsigned blockIdx;
        inliersH = kernel::ireduce_all<unsigned, af_max_t>(&blockIdx, inliers);

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
