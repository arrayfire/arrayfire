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
#include <kernel_headers/approx1.hpp>
#include <kernel_headers/approx2.hpp>
#include <kernel_headers/interp.hpp>
#include <math.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <string>
#include "config.hpp"
#include "interp.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int TX = 16;
static const int TY = 16;

static const int THREADS = 256;

template<typename Ty, typename Tp, int order>
std::string generateOptionsString() {
    ToNumStr<Ty> toNumStr;

    std::ostringstream options;
    options << " -D Ty=" << dtype_traits<Ty>::getName()
            << " -D Tp=" << dtype_traits<Tp>::getName()
            << " -D InterpInTy=" << dtype_traits<Ty>::getName()
            << " -D InterpValTy=" << dtype_traits<Ty>::getName()
            << " -D InterpPosTy=" << dtype_traits<Tp>::getName()
            << " -D ZERO=" << toNumStr(scalar<Ty>(0));

    if (static_cast<af_dtype>(dtype_traits<Ty>::af_type) == c32 ||
        static_cast<af_dtype>(dtype_traits<Ty>::af_type) == c64) {
        options << " -D IS_CPLX=1";
    } else {
        options << " -D IS_CPLX=0";
    }
    options << getTypeBuildDefinition<Ty>();

    options << " -D INTERP_ORDER=" << order;
    addInterpEnumOptions(options);

    return options.str();
}

///////////////////////////////////////////////////////////////////////////
// Wrapper functions
///////////////////////////////////////////////////////////////////////////
template<typename Ty, typename Tp, int order>
void approx1(Param yo, const Param yi, const Param xo, const int xdim,
             const Tp xi_beg, const Tp xi_step, const float offGrid,
             af_interp_type method) {
    std::string refName = std::string("approx1_kernel_") +
                          std::string(dtype_traits<Ty>::getName()) +
                          std::string(dtype_traits<Tp>::getName()) +
                          std::to_string(order);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::string options = generateOptionsString<Ty, Tp, order>();

        const char *ker_strs[] = {interp_cl, approx1_cl};
        const int ker_lens[]   = {interp_cl_len, approx1_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "approx1_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto approx1Op =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const Buffer, const KParam, const int, const Tp, const Tp,
                      const Ty, const int, const int, const int>(*entry.ker);

    NDRange local(THREADS, 1, 1);
    dim_t blocksPerMat = divup(yo.info.dims[0], local[0]);
    NDRange global(blocksPerMat * local[0] * yo.info.dims[1],
                   yo.info.dims[2] * yo.info.dims[3] * local[1]);

    // Passing bools to opencl kernels is not allowed
    bool batch =
        !(xo.info.dims[1] == 1 && xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    approx1Op(EnqueueArgs(getQueue(), global, local), *yo.data, yo.info,
              *yi.data, yi.info, *xo.data, xo.info, xdim, xi_beg, xi_step,
              scalar<Ty>(offGrid), blocksPerMat, (int)batch, (int)method);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ty, typename Tp, int order>
void approx2(Param zo, const Param zi, const Param xo, const int xdim,
             const Tp &xi_beg, const Tp &xi_step, const Param yo,
             const int ydim, const Tp &yi_beg, const Tp &yi_step,
             const float offGrid, af_interp_type method) {
    std::string refName = std::string("approx2_kernel_") +
                          std::string(dtype_traits<Ty>::getName()) +
                          std::string(dtype_traits<Tp>::getName()) +
                          std::to_string(order);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::string options = generateOptionsString<Ty, Tp, order>();

        const char *ker_strs[] = {interp_cl, approx2_cl};
        const int ker_lens[]   = {interp_cl_len, approx2_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "approx2_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto approx2Op =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const Buffer, const KParam, const int, const Buffer,
                      const KParam, const int, const Tp, const Tp, const Tp,
                      const Tp, const Ty, const int, const int, const int,
                      const int>(*entry.ker);

    NDRange local(TX, TY, 1);
    dim_t blocksPerMatX = divup(zo.info.dims[0], local[0]);
    dim_t blocksPerMatY = divup(zo.info.dims[1], local[1]);
    NDRange global(blocksPerMatX * local[0] * zo.info.dims[2],
                   blocksPerMatY * local[1] * zo.info.dims[3], 1);

    // Passing bools to opencl kernels is not allowed
    bool batch = !(xo.info.dims[2] == 1 && xo.info.dims[3] == 1);

    approx2Op(EnqueueArgs(getQueue(), global, local), *zo.data, zo.info,
              *zi.data, zi.info, *xo.data, xo.info, xdim, *yo.data, yo.info,
              ydim, xi_beg, xi_step, yi_beg, yi_step, scalar<Ty>(offGrid),
              blocksPerMatX, blocksPerMatY, (int)batch, (int)method);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
