/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/approx1.hpp>
#include <kernel_headers/approx2.hpp>
#include <kernel_headers/interp.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include <math.hpp>
#include "config.hpp"
#include "interp.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
static const int TX = 16;
static const int TY = 16;

static const int THREADS = 256;

template<typename Ty, typename Tp, int order>
std::string generateOptionsString()
{
    ToNumStr<Ty> toNumStr;

    std::ostringstream options;
    options << " -D Ty="          << dtype_traits<Ty>::getName()
        << " -D Tp="          << dtype_traits<Tp>::getName()
        << " -D InterpInTy="  << dtype_traits<Ty>::getName()
        << " -D InterpValTy=" << dtype_traits<Ty>::getName()
        << " -D InterpPosTy=" << dtype_traits<Tp>::getName()
        << " -D ZERO="        << toNumStr(scalar<Ty>(0));

    if((af_dtype) dtype_traits<Ty>::af_type == c32 ||
            (af_dtype) dtype_traits<Ty>::af_type == c64) {
        options << " -D IS_CPLX=1";
    } else {
        options << " -D IS_CPLX=0";
    }
    if (std::is_same<Ty, double>::value ||
            std::is_same<Ty, cdouble>::value) {
        options << " -D USE_DOUBLE";
    }

    options << " -D INTERP_ORDER=" << order;
    addInterpEnumOptions(options);

    return options.str();
}

///////////////////////////////////////////////////////////////////////////
// Wrapper functions
///////////////////////////////////////////////////////////////////////////
template <typename Ty, typename Tp, int order>
void approx1(Param out, const Param in, const Param xpos, const float offGrid,
             af_interp_type method)
{
    std::string refName = std::string("approx1_kernel_") +
        std::string(dtype_traits<Ty>::getName()) +
        std::string(dtype_traits<Tp>::getName()) +
        std::to_string(order);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::string options = generateOptionsString<Ty, Tp, order>();

        const char *ker_strs[] = {interp_cl, approx1_cl};
        const int   ker_lens[] = {interp_cl_len, approx1_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "approx1_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto approx1Op = KernelFunctor< Buffer, const KParam, const Buffer, const KParam,
                                    const Buffer, const KParam, const Ty,
                                    const int, const int, const int >(*entry.ker);

    NDRange local(THREADS, 1, 1);
    dim_t blocksPerMat = divup(out.info.dims[0], local[0]);
    NDRange global(blocksPerMat * local[0] * out.info.dims[1],
                   out.info.dims[2] * out.info.dims[3] * local[1], 1);

    // Passing bools to opencl kernels is not allowed
    bool batch = !(xpos.info.dims[1] == 1 && xpos.info.dims[2] == 1 && xpos.info.dims[3] == 1);

    approx1Op(EnqueueArgs(getQueue(), global, local),
              *out.data, out.info, *in.data, in.info,
              *xpos.data, xpos.info, scalar<Ty>(offGrid),
              blocksPerMat, (int)batch, (int)method);

    CL_DEBUG_FINISH(getQueue());
}

template <typename Ty, typename Tp, int order>
void approx2(Param out, const Param in, const Param xpos, const Param ypos,
             const float offGrid, af_interp_type method)
{
    std::string refName = std::string("approx2_kernel_") +
        std::string(dtype_traits<Ty>::getName()) +
        std::string(dtype_traits<Tp>::getName()) +
        std::to_string(order);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::string options = generateOptionsString<Ty, Tp, order>();

        const char *ker_strs[] = {interp_cl, approx2_cl};
        const int   ker_lens[] = {interp_cl_len, approx2_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "approx2_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto approx2Op = KernelFunctor< Buffer, const KParam, const Buffer, const KParam,
                                    const Buffer, const KParam, const Buffer, const KParam,
                                    const Ty, const int, const int, const int, const int >(*entry.ker);

    NDRange local(TX, TY, 1);
    dim_t blocksPerMatX = divup(out.info.dims[0], local[0]);
    dim_t blocksPerMatY = divup(out.info.dims[1], local[1]);
    NDRange global(blocksPerMatX * local[0] * out.info.dims[2],
                   blocksPerMatY * local[1] * out.info.dims[3], 1);

    // Passing bools to opencl kernels is not allowed
    bool batch = !(xpos.info.dims[2] == 1 && xpos.info.dims[3] == 1);

    approx2Op(EnqueueArgs(getQueue(), global, local),
              *out.data, out.info, *in.data, in.info,
              *xpos.data, xpos.info, *ypos.data, ypos.info,
              scalar<Ty>(offGrid), blocksPerMatX, blocksPerMatY, (int)batch, (int)method);

    CL_DEBUG_FINISH(getQueue());
}
}
}
