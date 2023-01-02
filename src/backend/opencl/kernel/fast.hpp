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
#include <kernel_headers/fast.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T>
void fast(const unsigned arc_length, unsigned *out_feat, Param &x_out,
          Param &y_out, Param &score_out, Param in, const float thr,
          const float feature_ratio, const unsigned edge, const bool nonmax) {
    constexpr int FAST_THREADS_X        = 16;
    constexpr int FAST_THREADS_Y        = 16;
    constexpr int FAST_THREADS_NONMAX_X = 32;
    constexpr int FAST_THREADS_NONMAX_Y = 8;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(arc_length),
        TemplateArg(nonmax),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ARC_LENGTH, arc_length),
        DefineKeyValue(NONMAX, static_cast<unsigned>(nonmax)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto locate =
        common::getKernel("locate_features", {{fast_cl_src}}, targs, options);
    auto nonMax =
        common::getKernel("non_max_counts", {{fast_cl_src}}, targs, options);
    auto getFeat =
        common::getKernel("get_features", {{fast_cl_src}}, targs, options);

    const unsigned max_feat =
        ceil(in.info.dims[0] * in.info.dims[1] * feature_ratio);

    // Matrix containing scores for detected features, scores are stored in the
    // same coordinates as features, dimensions should be equal to in.
    cl::Buffer *d_score =
        bufferAlloc(in.info.dims[0] * in.info.dims[1] * sizeof(float));
    getQueue().enqueueFillBuffer(
        *d_score, 0.0F, 0, in.info.dims[0] * in.info.dims[1] * sizeof(float));

    cl::Buffer *d_flags = d_score;
    if (nonmax) {
        d_flags =
            bufferAlloc(in.info.dims[0] * in.info.dims[1] * sizeof(float));
    }

    const int blk_x = divup(in.info.dims[0] - edge * 2, FAST_THREADS_X);
    const int blk_y = divup(in.info.dims[1] - edge * 2, FAST_THREADS_Y);

    // Locate features kernel sizes
    const cl::NDRange local(FAST_THREADS_X, FAST_THREADS_Y);
    const cl::NDRange global(blk_x * FAST_THREADS_X, blk_y * FAST_THREADS_Y);

    locate(cl::EnqueueArgs(getQueue(), global, local), *in.data, in.info,
           *d_score, thr, edge,
           cl::Local((FAST_THREADS_X + 6) * (FAST_THREADS_Y + 6) * sizeof(T)));
    CL_DEBUG_FINISH(getQueue());

    const int blk_nonmax_x = divup(in.info.dims[0], 64);
    const int blk_nonmax_y = divup(in.info.dims[1], 64);

    // Nonmax kernel sizes
    const cl::NDRange local_nonmax(FAST_THREADS_NONMAX_X,
                                   FAST_THREADS_NONMAX_Y);
    const cl::NDRange global_nonmax(blk_nonmax_x * FAST_THREADS_NONMAX_X,
                                    blk_nonmax_y * FAST_THREADS_NONMAX_Y);

    cl::Buffer *d_total = bufferAlloc(sizeof(unsigned));
    getQueue().enqueueFillBuffer(*d_total, 0U, 0, sizeof(unsigned));

    // size_t *global_nonmax_dims = global_nonmax();
    size_t blocks_sz = blk_nonmax_x * FAST_THREADS_NONMAX_X * blk_nonmax_y *
                       FAST_THREADS_NONMAX_Y * sizeof(unsigned);
    cl::Buffer *d_counts  = bufferAlloc(blocks_sz);
    cl::Buffer *d_offsets = bufferAlloc(blocks_sz);

    nonMax(cl::EnqueueArgs(getQueue(), global_nonmax, local_nonmax), *d_counts,
           *d_offsets, *d_total, *d_flags, *d_score, in.info, edge);
    CL_DEBUG_FINISH(getQueue());

    unsigned total;
    getQueue().enqueueReadBuffer(*d_total, CL_TRUE, 0, sizeof(unsigned),
                                 &total);
    total = total < max_feat ? total : max_feat;

    if (total > 0) {
        size_t out_sz  = total * sizeof(float);
        x_out.data     = bufferAlloc(out_sz);
        y_out.data     = bufferAlloc(out_sz);
        score_out.data = bufferAlloc(out_sz);

        getFeat(cl::EnqueueArgs(getQueue(), global_nonmax, local_nonmax),
                *x_out.data, *y_out.data, *score_out.data, *d_flags, *d_counts,
                *d_offsets, in.info, total, edge);
        CL_DEBUG_FINISH(getQueue());
    }

    *out_feat = total;

    x_out.info.dims[0]        = total;
    x_out.info.strides[0]     = 1;
    y_out.info.dims[0]        = total;
    y_out.info.strides[0]     = 1;
    score_out.info.dims[0]    = total;
    score_out.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        x_out.info.dims[k]        = 1;
        x_out.info.strides[k]     = total;
        y_out.info.dims[k]        = 1;
        y_out.info.strides[k]     = total;
        score_out.info.dims[k]    = 1;
        score_out.info.strides[k] = total;
    }

    bufferFree(d_score);
    if (nonmax) bufferFree(d_flags);
    bufferFree(d_total);
    bufferFree(d_counts);
    bufferFree(d_offsets);
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
