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
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/scan_dim.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>

#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
template<typename Ti, typename To, af_op_t op, bool inclusive_scan>
static Kernel get_scan_dim_kernels(int kerIdx, int dim, bool isFinalPass,
                                   uint threads_y) {
    std::string ref_name =
        std::string("scan_") + std::to_string(dim) + std::string("_") +
        std::to_string(isFinalPass) + std::string("_") +
        std::string(dtype_traits<Ti>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_y) +
        std::string("_") + std::to_string(int(inclusive_scan));

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Ti=" << dtype_traits<Ti>::getName() << " -D T=To"
                << " -D kDim=" << dim << " -D DIMY=" << threads_y
                << " -D THREADS_X=" << THREADS_X
                << " -D init=" << toNumStr(Binary<To, op>::init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<Ti>()
                << " -D isFinalPass=" << (int)(isFinalPass)
                << " -D inclusive_scan=" << inclusive_scan;
        options << getTypeBuildDefinition<Ti>();

        const char *ker_strs[] = {ops_cl, scan_dim_cl};
        const int ker_lens[]   = {ops_cl_len, scan_dim_cl_len};
        cl::Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel[2];

        entry.ker[0] = Kernel(*entry.prog, "scan_dim_kernel");
        entry.ker[1] = Kernel(*entry.prog, "bcast_dim_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    return entry.ker[kerIdx];
}

template<typename Ti, typename To, af_op_t op, bool inclusive_scan>
static void scan_dim_launcher(Param out, Param tmp, const Param in, int dim,
                              bool isFinalPass, uint threads_y,
                              const uint groups_all[4]) {
    Kernel ker = get_scan_dim_kernels<Ti, To, op, inclusive_scan>(
        0, dim, isFinalPass, threads_y);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    auto scanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                                uint, uint, uint, uint>(ker);

    scanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *tmp.data, tmp.info, *in.data, in.info, groups_all[0], groups_all[1],
           groups_all[dim], lim);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op, bool inclusive_scan>
static void bcast_dim_launcher(Param out, Param tmp, int dim, bool isFinalPass,
                               uint threads_y, const uint groups_all[4]) {
    Kernel ker = get_scan_dim_kernels<Ti, To, op, inclusive_scan>(
        1, dim, isFinalPass, threads_y);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

    auto bcastOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, uint, uint, uint, uint>(
            ker);

    bcastOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
            *tmp.data, tmp.info, groups_all[0], groups_all[1], groups_all[dim],
            lim);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op, bool inclusive_scan = true>
static void scan_dim(Param out, const Param in, int dim) {
    uint threads_y = std::min(THREADS_Y, nextpow2(out.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {divup((uint)out.info.dims[0], threads_x),
                         (uint)out.info.dims[1], (uint)out.info.dims[2],
                         (uint)out.info.dims[3]};

    groups_all[dim] = divup(out.info.dims[dim], threads_y * REPEAT);

    if (groups_all[dim] == 1) {
        scan_dim_launcher<Ti, To, op, inclusive_scan>(out, out, in, dim, true,
                                                      threads_y, groups_all);
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

        scan_dim_launcher<Ti, To, op, inclusive_scan>(out, tmp, in, dim, false,
                                                      threads_y, groups_all);

        int gdim        = groups_all[dim];
        groups_all[dim] = 1;

        if (op == af_notzero_t) {
            scan_dim_launcher<To, To, af_add_t, true>(tmp, tmp, tmp, dim, true,
                                                      threads_y, groups_all);
        } else {
            scan_dim_launcher<To, To, op, true>(tmp, tmp, tmp, dim, true,
                                                threads_y, groups_all);
        }

        groups_all[dim] = gdim;
        bcast_dim_launcher<To, To, op, inclusive_scan>(out, tmp, dim, true,
                                                       threads_y, groups_all);
        bufferFree(tmp.data);
    }
}
}  // namespace kernel
}  // namespace opencl
