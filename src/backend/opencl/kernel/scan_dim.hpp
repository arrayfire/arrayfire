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
#include <common/Binary.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/scan_dim.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename Ti, typename To, af_op_t op>
static opencl::Kernel getScanDimKernel(const std::string key, int dim,
                                       bool isFinalPass, uint threads_y,
                                       bool inclusiveScan) {
    using std::string;
    using std::vector;

    ToNumStr<To> toNumStr;
    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(dim),
        TemplateArg(isFinalPass),
        TemplateArg(op),
        TemplateArg(threads_y),
        TemplateArg(inclusiveScan),
    };
    vector<string> compileOpts = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(kDim, dim),
        DefineKeyValue(DIMY, threads_y),
        DefineValue(THREADS_X),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        DefineKeyValue(IS_FINAL_PASS, (isFinalPass ? 1 : 0)),
        DefineKeyValue(INCLUSIVE_SCAN, inclusiveScan),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    return common::getKernel(key, {{ops_cl_src, scan_dim_cl_src}}, tmpltArgs,
                             compileOpts);
}

template<typename Ti, typename To, af_op_t op>
static void scanDimLauncher(Param out, Param tmp, const Param in, int dim,
                            bool isFinalPass, uint threads_y,
                            const uint groups_all[4], bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanDimKernel<Ti, To, op>("scanDim", dim, isFinalPass,
                                             threads_y, inclusiveScan);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *tmp.data,
         tmp.info, *in.data, in.info, groups_all[0], groups_all[1],
         groups_all[dim], lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
static void bcastDimLauncher(Param out, Param tmp, int dim, bool isFinalPass,
                             uint threads_y, const uint groups_all[4],
                             const bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto bcast = getScanDimKernel<Ti, To, op>("bcastDim", dim, isFinalPass,
                                              threads_y, inclusiveScan);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    bcast(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *tmp.data, tmp.info, groups_all[0], groups_all[1], groups_all[dim],
          lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
static void scanDim(Param out, const Param in, const int dim,
                    const bool inclusiveScan = true) {
    uint threads_y = std::min(THREADS_Y, nextpow2(out.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {divup((uint)out.info.dims[0], threads_x),
                         (uint)out.info.dims[1], (uint)out.info.dims[2],
                         (uint)out.info.dims[3]};

    groups_all[dim] = divup(out.info.dims[dim], threads_y * REPEAT);

    if (groups_all[dim] == 1) {
        scanDimLauncher<Ti, To, op>(out, out, in, dim, true, threads_y,
                                    groups_all, inclusiveScan);
    } else {
        Param tmp = out;

        tmp.info.dims[dim]  = groups_all[dim];
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++) {
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];
        }

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
        // FIXME: Do I need to free this ?
        tmp.data = bufferAlloc(tmp_elements * sizeof(To));

        scanDimLauncher<Ti, To, op>(out, tmp, in, dim, false, threads_y,
                                    groups_all, inclusiveScan);

        int gdim        = groups_all[dim];
        groups_all[dim] = 1;

        if (op == af_notzero_t) {
            scanDimLauncher<To, To, af_add_t>(tmp, tmp, tmp, dim, true,
                                              threads_y, groups_all, true);
        } else {
            scanDimLauncher<To, To, op>(tmp, tmp, tmp, dim, true, threads_y,
                                        groups_all, true);
        }

        groups_all[dim] = gdim;
        bcastDimLauncher<To, To, op>(out, tmp, dim, true, threads_y, groups_all,
                                     inclusiveScan);
        bufferFree(tmp.data);
    }
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
