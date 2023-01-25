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
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel/scan_first.hpp>
#include <kernel_headers/where.hpp>
#include <math.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
static void get_out_idx(cl::Buffer *out_data, Param &otmp, Param &rtmp,
                        Param &in, uint threads_x, uint groups_x,
                        uint groups_y) {
    using cl::EnqueueArgs;
    using cl::NDRange;
    using std::string;
    using std::vector;

    ToNumStr<T> toNumStr;
    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };
    vector<string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ZERO, toNumStr(scalar<T>(0))),
        DefineKeyValue(CPLX, iscplx<T>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto getIdx = common::getKernel("get_out_idx", {{where_cl_src}}, tmpltArgs,
                                    compileOpts);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(local[0] * groups_x * in.info.dims[2],
                   local[1] * groups_y * in.info.dims[3]);

    uint lim = divup(otmp.info.dims[0], (threads_x * groups_x));

    getIdx(EnqueueArgs(getQueue(), global, local), *out_data, *otmp.data,
           otmp.info, *rtmp.data, rtmp.info, *in.data, in.info, groups_x,
           groups_y, lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
static void where(Param &out, Param &in) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param rtmp;
    Param otmp;

    rtmp.info.dims[0] = groups_x;
    otmp.info.dims[0] = in.info.dims[0];

    rtmp.info.strides[0] = 1;
    otmp.info.strides[0] = 1;

    rtmp.info.offset = 0;
    otmp.info.offset = 0;

    for (int k = 1; k < 4; k++) {
        rtmp.info.dims[k]    = in.info.dims[k];
        rtmp.info.strides[k] = rtmp.info.strides[k - 1] * rtmp.info.dims[k - 1];

        otmp.info.dims[k]    = in.info.dims[k];
        otmp.info.strides[k] = otmp.info.strides[k - 1] * otmp.info.dims[k - 1];
    }

    int rtmp_elements = rtmp.info.strides[3] * rtmp.info.dims[3];
    rtmp.data         = bufferAlloc(rtmp_elements * sizeof(uint));

    int otmp_elements = otmp.info.strides[3] * otmp.info.dims[3];
    otmp.data         = bufferAlloc(otmp_elements * sizeof(uint));

    scanFirstLauncher<T, uint, af_notzero_t>(otmp, rtmp, in, false, groups_x,
                                             groups_y, threads_x);

    // Linearize the dimensions and perform scan
    Param ltmp        = rtmp;
    ltmp.info.offset  = 0;
    ltmp.info.dims[0] = rtmp_elements;
    for (int k = 1; k < 4; k++) {
        ltmp.info.dims[k]    = 1;
        ltmp.info.strides[k] = rtmp_elements;
    }

    scanFirst<uint, uint, af_add_t>(ltmp, ltmp);

    // Get output size and allocate output
    uint total;
    getQueue().enqueueReadBuffer(*rtmp.data, CL_TRUE,
                                 sizeof(uint) * (rtmp_elements - 1),
                                 sizeof(uint), &total);

    out.data = bufferAlloc(total * sizeof(uint));

    out.info.dims[0]    = total;
    out.info.strides[0] = 1;
    for (int k = 1; k < 4; k++) {
        out.info.dims[k]    = 1;
        out.info.strides[k] = total;
    }

    if (total > 0)
        get_out_idx<T>(out.data, otmp, rtmp, in, threads_x, groups_x, groups_y);

    bufferFree(rtmp.data);
    bufferFree(otmp.data);
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
