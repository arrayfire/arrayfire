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
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/scan_first_by_key.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <map>
#include <mutex>
#include <string>
#include "config.hpp"
#include "names.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {

template<typename Ti, typename Tk, typename To, af_op_t op, bool inclusive_scan>
static Kernel get_scan_first_kernels(int kerIdx, bool calculateFlags,
                                     uint threads_x) {
    std::string ref_name =
        std::string("scan_0_") + std::string("_") +
        std::to_string(calculateFlags) + std::string("_") +
        std::string(dtype_traits<Ti>::getName()) + std::string("_") +
        std::string(dtype_traits<Tk>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_x) +
        std::string("_") + std::to_string(int(inclusive_scan));

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        const uint threads_y       = THREADS_PER_GROUP / threads_x;
        const uint SHARED_MEM_SIZE = THREADS_PER_GROUP;

        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Ti=" << dtype_traits<Ti>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x << " -D DIMY=" << threads_y
                << " -D SHARED_MEM_SIZE=" << SHARED_MEM_SIZE
                << " -D init=" << toNumStr(Binary<To, op>::init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<Ti>()
                << " -D calculateFlags=" << calculateFlags
                << " -D inclusive_scan=" << inclusive_scan;
        options << getTypeBuildDefinition<Ti>();

        const char *ker_strs[] = {ops_cl, scan_first_by_key_cl};
        const int ker_lens[]   = {ops_cl_len, scan_first_by_key_cl_len};
        cl::Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel[3];

        entry.ker[0] = Kernel(*entry.prog, "scan_first_by_key_final_kernel");
        entry.ker[1] = Kernel(*entry.prog, "scan_first_by_key_nonfinal_kernel");
        entry.ker[2] = Kernel(*entry.prog, "bcast_first_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    return entry.ker[kerIdx];
}

template<typename Ti, typename Tk, typename To, af_op_t op,
         bool inclusive_scan = true>
static void scan_first_nonfinal_launcher(Param &out, Param &tmp, Param &tmpflg,
                                         Param &tmpid, const Param &in,
                                         const Param &key, const uint groups_x,
                                         const uint groups_y,
                                         const uint threads_x) {
    Kernel ker = get_scan_first_kernels<Ti, Tk, To, op, inclusive_scan>(
        1, false, threads_x);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    auto scanOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam, Buffer,
                      KParam, Buffer, KParam, Buffer, KParam, uint, uint, uint>(
            ker);

    scanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *tmp.data, tmp.info, *tmpflg.data, tmpflg.info, *tmpid.data,
           tmpid.info, *in.data, in.info, *key.data, key.info, groups_x,
           groups_y, lim);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op,
         bool inclusive_scan = true>
static void scan_first_final_launcher(Param &out, const Param &in,
                                      const Param &key,
                                      const bool calculateFlags,
                                      const uint groups_x, const uint groups_y,
                                      const uint threads_x) {
    Kernel ker = get_scan_first_kernels<Ti, Tk, To, op, inclusive_scan>(
        0, calculateFlags, threads_x);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    auto scanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                                uint, uint, uint>(ker);

    scanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *in.data, in.info, *key.data, key.info, groups_x, groups_y, lim);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op, bool inclusive_scan>
static void bcast_first_launcher(Param &out, Param &tmp, Param &tmpid,
                                 const uint groups_x, const uint groups_y,
                                 const uint threads_x) {
    Kernel ker = get_scan_first_kernels<Ti, Tk, To, op, inclusive_scan>(
        2, false, threads_x);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * out.info.dims[2] * local[0],
                   groups_y * out.info.dims[3] * local[1]);

    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    auto bcastOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                                 uint, uint, uint>(ker);

    bcastOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            *tmp.data, tmp.info, *tmpid.data, tmpid.info, groups_x, groups_y,
            lim);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op, bool inclusive_scan>
void scan_first(Param &out, const Param &in, const Param &key) {
    uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(out.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(out.info.dims[1], threads_y);

    if (groups_x == 1) {
        scan_first_final_launcher<Ti, Tk, To, op, inclusive_scan>(
            out, in, key, true, groups_x, groups_y, threads_x);

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

        scan_first_nonfinal_launcher<Ti, Tk, To, op, inclusive_scan>(
            out, tmp, tmpflg, tmpid, in, key, groups_x, groups_y, threads_x);

        if (op == af_notzero_t) {
            scan_first_final_launcher<To, char, To, af_add_t, true>(
                tmp, tmp, tmpflg, false, 1, groups_y, threads_x);
        } else {
            scan_first_final_launcher<To, char, To, op, true>(
                tmp, tmp, tmpflg, false, 1, groups_y, threads_x);
        }

        bcast_first_launcher<To, Tk, To, op, inclusive_scan>(
            out, tmp, tmpid, groups_x, groups_y, threads_x);

        bufferFree(tmp.data);
        bufferFree(tmpflg.data);
        bufferFree(tmpid.data);
    }
}

}  // namespace kernel

#define INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, Ti, Tk, To)   \
    template void scan_first<Ti, Tk, To, ROp, true>(     \
        Param & out, const Param &in, const Param &key); \
    template void scan_first<Ti, Tk, To, ROp, false>(    \
        Param & out, const Param &in, const Param &key);

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
