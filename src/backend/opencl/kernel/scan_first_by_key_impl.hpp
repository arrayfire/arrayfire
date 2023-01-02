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
#include <kernel_headers/ops.hpp>
#include <kernel_headers/scan_first_by_key.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename Ti, typename Tk, typename To, af_op_t op>
static opencl::Kernel getScanFirstKernel(const std::string key,
                                         bool calculateFlags, uint threads_x,
                                         const bool inclusiveScan) {
    using std::string;
    using std::vector;

    const uint threads_y       = THREADS_PER_GROUP / threads_x;
    const uint SHARED_MEM_SIZE = THREADS_PER_GROUP;
    ToNumStr<To> toNumStr;
    vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateTypename<Tk>(),
        TemplateArg(calculateFlags),
        TemplateArg(op),
        TemplateArg(threads_x),
        TemplateArg(inclusiveScan),
    };
    vector<string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIMY, threads_y),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineValue(SHARED_MEM_SIZE),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        DefineKeyValue(calculateFlags, (calculateFlags ? 1 : 0)),
        DefineKeyValue(INCLUSIVE_SCAN, inclusiveScan),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    return common::getKernel(key, {{ops_cl_src, scan_first_by_key_cl_src}},
                             tmpltArgs, compileOpts);
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scanFirstByKeyNonfinalLauncher(
    Param &out, Param &tmp, Param &tmpflg, Param &tmpid, const Param &in,
    const Param &key, const uint groups_x, const uint groups_y,
    const uint threads_x, const bool inclusiveScan = true) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanFirstKernel<Ti, Tk, To, op>(
        "scanFirstByKeyNonfinal", false, threads_x, inclusiveScan);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *tmp.data,
         tmp.info, *tmpflg.data, tmpflg.info, *tmpid.data, tmpid.info, *in.data,
         in.info, *key.data, key.info, groups_x, groups_y, lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void scanFirstByKeyFinalLauncher(
    Param &out, const Param &in, const Param &key, const bool calculateFlags,
    const uint groups_x, const uint groups_y, const uint threads_x,
    const bool inclusiveScan = true) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto scan = getScanFirstKernel<Ti, Tk, To, op>(
        "scanFirstByKeyFinal", calculateFlags, threads_x, inclusiveScan);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    scan(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data,
         in.info, *key.data, key.info, groups_x, groups_y, lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
static void bcastFirstByKeyLauncher(Param &out, Param &tmp, Param &tmpid,
                                    const uint groups_x, const uint groups_y,
                                    const uint threads_x, bool inclusiveScan) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    auto bcast = getScanFirstKernel<Ti, Tk, To, op>("bcastFirstByKey", false,
                                                    threads_x, inclusiveScan);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    bcast(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
          *tmp.data, tmp.info, *tmpid.data, tmpid.info, groups_x, groups_y,
          lim);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void scanFirstByKey(Param &out, const Param &in, const Param &key,
                    const bool inclusiveScan) {
    uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(out.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(out.info.dims[1], threads_y);

    if (groups_x == 1) {
        scanFirstByKeyFinalLauncher<Ti, Tk, To, op>(
            out, in, key, true, groups_x, groups_y, threads_x, inclusiveScan);

    } else {
        Param tmp           = out;
        tmp.info.dims[0]    = groups_x;
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++) {
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];
        }
        Param tmpflg = tmp;
        Param tmpid  = tmp;

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];

        tmp.data    = bufferAlloc(tmp_elements * sizeof(To));
        tmpflg.data = bufferAlloc(tmp_elements * sizeof(char));
        tmpid.data  = bufferAlloc(tmp_elements * sizeof(int));

        scanFirstByKeyNonfinalLauncher<Ti, Tk, To, op>(
            out, tmp, tmpflg, tmpid, in, key, groups_x, groups_y, threads_x,
            inclusiveScan);

        if (op == af_notzero_t) {
            scanFirstByKeyFinalLauncher<To, char, To, af_add_t>(
                tmp, tmp, tmpflg, false, 1, groups_y, threads_x, true);
        } else {
            scanFirstByKeyFinalLauncher<To, char, To, op>(
                tmp, tmp, tmpflg, false, 1, groups_y, threads_x, true);
        }

        bcastFirstByKeyLauncher<To, Tk, To, op>(
            out, tmp, tmpid, groups_x, groups_y, threads_x, inclusiveScan);

        bufferFree(tmp.data);
        bufferFree(tmpflg.data);
        bufferFree(tmpid.data);
    }
}
}  // namespace kernel

#define INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, Ti, Tk, To) \
    template void scanFirstByKey<Ti, Tk, To, ROp>(     \
        Param & out, const Param &in, const Param &key, const bool);

#define INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, Tk)         \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_FIRST_BY_KEY_OP(ROp)      \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, int)  \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uint) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, intl) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uintl)
}  // namespace opencl
}  // namespace arrayfire
