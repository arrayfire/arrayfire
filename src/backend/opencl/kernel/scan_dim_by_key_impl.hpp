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
#include <kernel_headers/scan_dim_by_key.hpp>
#include <memory.hpp>
#include <optypes.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename Ti, typename Tk, typename To, af_op_t op>
static opencl::Kernel getScanDimKernel(const std::string key, int dim,
                                       bool calculateFlags, uint threads_y,
                                       bool inclusiveScan) {
    using std::string;
    using std::vector;

    ToNumStr<To> toNumStr;
    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(),      TemplateTypename<To>(),
        TemplateTypename<Tk>(),      TemplateArg(dim),
        TemplateArg(calculateFlags), TemplateArg(op),
        TemplateArg(threads_y),      TemplateArg(inclusiveScan),
    };
    vector<string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(kDim, dim),
        DefineKeyValue(DIMY, threads_y),
        DefineValue(THREADS_X),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        DefineKeyValue(calculateFlags, (calculateFlags ? 1 : 0)),
        DefineKeyValue(INCLUSIVE_SCAN, inclusiveScan),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    return common::getKernel(key, {{ops_cl_src, scan_dim_by_key_cl_src}},
                             tmpltArgs, compileOpts);
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scanDimNonfinalLauncher(Param out, Param tmp, Param tmpflg,
                                    Param tmpid, const Param in,
                                    const Param key, int dim, uint threads_y,
                                    const uint groups_all[4],
                                    bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanDimKernel<Ti, Tk, To, op>(
        "scanDimByKeyNonfinal", dim, false, threads_y, inclusiveScan);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *tmp.data,
         tmp.info, *tmpflg.data, tmpflg.info, *tmpid.data, tmpid.info, *in.data,
         in.info, *key.data, key.info, groups_all[0], groups_all[1],
         groups_all[dim], lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scanDimFinalLauncher(Param out, const Param in, const Param key,
                                 int dim, const bool calculateFlags,
                                 uint threads_y, const uint groups_all[4],
                                 bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanDimKernel<Ti, Tk, To, op>(
        "scanDimByKeyFinal", dim, calculateFlags, threads_y, inclusiveScan);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
         in.info, *key.data, key.info, groups_all[0], groups_all[1],
         groups_all[dim], lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void bcastDimLauncher(Param out, Param tmp, Param tmpid, int dim,
                             uint threads_y, const uint groups_all[4],
                             bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto bcast = getScanDimKernel<Ti, Tk, To, op>("bcastDimByKey", dim, false,
                                                  threads_y, inclusiveScan);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    bcast(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *tmp.data, tmp.info, *tmpid.data, tmpid.info, groups_all[0],
          groups_all[1], groups_all[dim], lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void scanDimByKey(Param out, const Param in, const Param key, int dim,
                  const bool inclusiveScan) {
    uint threads_y = std::min(THREADS_Y, nextpow2(out.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {divup((uint)out.info.dims[0], threads_x),
                         (uint)out.info.dims[1], (uint)out.info.dims[2],
                         (uint)out.info.dims[3]};

    groups_all[dim] = divup(out.info.dims[dim], threads_y * REPEAT);

    if (groups_all[dim] == 1) {
        scanDimFinalLauncher<Ti, Tk, To, op>(out, in, key, dim, true, threads_y,
                                             groups_all, inclusiveScan);
    } else {
        Param tmp = out;

        tmp.info.dims[dim]  = groups_all[dim];
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++) {
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];
        }
        Param tmpflg = tmp;
        Param tmpid  = tmp;

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
        // FIXME: Do I need to free this ?
        tmp.data    = bufferAlloc(tmp_elements * sizeof(To));
        tmpflg.data = bufferAlloc(tmp_elements * sizeof(char));
        tmpid.data  = bufferAlloc(tmp_elements * sizeof(int));

        scanDimNonfinalLauncher<Ti, Tk, To, op>(out, tmp, tmpflg, tmpid, in,
                                                key, dim, threads_y, groups_all,
                                                inclusiveScan);

        int gdim        = groups_all[dim];
        groups_all[dim] = 1;

        if (op == af_notzero_t) {
            scanDimFinalLauncher<To, char, To, af_add_t>(
                tmp, tmp, tmpflg, dim, false, threads_y, groups_all, true);
        } else {
            scanDimFinalLauncher<To, char, To, op>(tmp, tmp, tmpflg, dim, false,
                                                   threads_y, groups_all, true);
        }

        groups_all[dim] = gdim;
        bcastDimLauncher<To, Tk, To, op>(out, tmp, tmpid, dim, threads_y,
                                         groups_all, inclusiveScan);
        bufferFree(tmp.data);
        bufferFree(tmpflg.data);
        bufferFree(tmpid.data);
    }
}
}  // namespace kernel

#define INSTANTIATE_SCAN_DIM_BY_KEY(ROp, Ti, Tk, To) \
    template void scanDimByKey<Ti, Tk, To, ROp>(     \
        Param out, const Param in, const Param key, int dim, const bool);

#define INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, Tk)         \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_DIM_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_DIM_BY_KEY_OP(ROp)      \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, int)  \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, uint) \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, intl) \
    INSTANTIATE_SCAN_DIM_BY_KEY_TYPES(ROp, uintl)
}  // namespace opencl
}  // namespace arrayfire
