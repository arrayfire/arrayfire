/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#include <kernel_headers/convolve.hpp>
#include <kernel_headers/ops.hpp>

#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel/names.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
static const int THREADS = 256;

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const int CUBE_X = 8;
static const int CUBE_Y = 8;
static const int CUBE_Z = 4;

struct conv_kparam_t {
    NDRange global;
    NDRange local;
    size_t loc_size;
    int nBBS0;
    int nBBS1;
    bool outHasNoOffset;
    bool inHasNoOffset;
    bool launchMoreBlocks;
    int o[3];
    int s[3];
    cl::Buffer* impulse;
};

template <typename T>
void prepareKernelArgs(conv_kparam_t& param, dim_t* oDims, const dim_t* fDims,
                       int baseDim) {
    int batchDims[4] = {1, 1, 1, 1};
    for (int i = baseDim; i < 4; ++i) {
        batchDims[i] = (param.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (baseDim == 1) {
        param.local  = NDRange(THREADS, 1);
        param.nBBS0  = divup(oDims[0], THREADS);
        param.nBBS1  = batchDims[2];
        param.global = NDRange(param.nBBS0 * THREADS * batchDims[1],
                               param.nBBS1 * batchDims[3]);
        param.loc_size = (THREADS + 2 * (fDims[0] - 1)) * sizeof(T);
    } else if (baseDim == 2) {
        param.local  = NDRange(THREADS_X, THREADS_Y);
        param.nBBS0  = divup(oDims[0], THREADS_X);
        param.nBBS1  = divup(oDims[1], THREADS_Y);
        param.global = NDRange(param.nBBS0 * THREADS_X * batchDims[2],
                               param.nBBS1 * THREADS_Y * batchDims[3]);
    } else if (baseDim == 3) {
        param.local  = NDRange(CUBE_X, CUBE_Y, CUBE_Z);
        param.nBBS0  = divup(oDims[0], CUBE_X);
        param.nBBS1  = divup(oDims[1], CUBE_Y);
        int blk_z    = divup(oDims[2], CUBE_Z);
        param.global = NDRange(param.nBBS0 * CUBE_X * batchDims[3],
                               param.nBBS1 * CUBE_Y, blk_z * CUBE_Z);
        param.loc_size = (CUBE_X + 2 * (fDims[0] - 1)) *
                         (CUBE_Y + 2 * (fDims[1] - 1)) *
                         (CUBE_Z + 2 * (fDims[2] - 1)) * sizeof(T);
    }
}

template <typename T, typename aT, int bDim, bool expand>
void convNHelper(const conv_kparam_t& param, Param& out, const Param& signal,
                 const Param& filter) {
    std::string ref_name = std::string("convolveND_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string(dtype_traits<aT>::getName()) +
                           std::to_string(bDim) + std::to_string(expand);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D Ti=" << dtype_traits<T>::getName()
                << " -D To=" << dtype_traits<aT>::getName()
                << " -D accType=" << dtype_traits<aT>::getName()
                << " -D BASE_DIM=" << bDim << " -D EXPAND=" << expand << " -D "
                << binOpName<af_mul_t>();

        if ((af_dtype)dtype_traits<T>::af_type == c32 ||
            (af_dtype)dtype_traits<T>::af_type == c64) {
            options << " -D CPLX=1";
        } else {
            options << " -D CPLX=0";
        }
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {ops_cl, convolve_cl};
        const int ker_lens[]   = {ops_cl_len, convolve_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "convolve");

        addKernelToCache(device, ref_name, entry);
    }

    auto convOp = cl::KernelFunctor<Buffer, KParam, Buffer, KParam,
                                    cl::LocalSpaceArg, Buffer, KParam, int, int,
                                    int, int, int, int, int, int>(*entry.ker);

    convOp(EnqueueArgs(getQueue(), param.global, param.local), *out.data,
           out.info, *signal.data, signal.info, cl::Local(param.loc_size),
           *param.impulse, filter.info, param.nBBS0, param.nBBS1, param.o[0],
           param.o[1], param.o[2], param.s[0], param.s[1], param.s[2]);
}

template <typename T, typename aT, bool expand>
void conv1(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);

template <typename T, typename aT, bool expand>
void conv2(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);

template <typename T, typename aT, bool expand>
void conv3(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);
}  // namespace kernel
}  // namespace opencl
