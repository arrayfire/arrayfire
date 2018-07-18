/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/morph.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <memory.hpp>
#include <ops.hpp>
#include <type_util.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

static const int CUBE_X    =  8;
static const int CUBE_Y    =  8;
static const int CUBE_Z    =  4;

template<typename T, bool isDilation, int SeLength>
std::string generateOptionsString()
{
    ToNumStr<T> toNumStr;
    T init = isDilation ? Binary<T, af_max_t>::init() : Binary<T, af_min_t>::init();
    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
        << " -D isDilation="<< isDilation
        << " -D init=" << toNumStr(init)
        << " -D SeLength=" << SeLength;
    if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
        options << " -D USE_DOUBLE";
    return options.str();
}

template<typename T, bool isDilation, int SeLength=0>
void morph(Param out, const Param in, const Param mask, int windLen=0)
{
    std::string refName = std::string("morph_") +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(isDilation) + std::to_string(SeLength);

    windLen = (SeLength>0 ? SeLength : windLen);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::string options = generateOptionsString<T, isDilation, SeLength>();
        const char* ker_strs[] = {morph_cl};
        const int   ker_lens[] = {morph_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "morph");
        addKernelToCache(device, refName, entry);
    }

    auto morphOp = KernelFunctor< Buffer, KParam, Buffer, KParam, Buffer, cl::LocalSpaceArg,
                                  int, int, int >(*entry.ker);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);
    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * THREADS_X * in.info.dims[2], blk_y * THREADS_Y * in.info.dims[3]);

    // copy mask/filter to constant memory
    cl_int se_size   = sizeof(T)*windLen*windLen;
    auto mBuff = memAlloc<T>(windLen*windLen);
    getQueue().enqueueCopyBuffer(*mask.data, *mBuff, 0, 0, se_size);

    // calculate shared memory size
    const int padding = (windLen%2==0 ? (windLen-1) : (2*(windLen/2)));
    const int locLen  = THREADS_X + padding + 1;
    const int locSize = locLen * (THREADS_Y+padding);

    morphOp(EnqueueArgs(getQueue(), global, local),
            *out.data, out.info, *in.data, in.info, *mBuff,
            cl::Local(locSize*sizeof(T)), blk_x, blk_y, windLen);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T, bool isDilation, int SeLength>
void morph3d(Param       out,
        const Param      in,
        const Param      mask)
{
    std::string refName = std::string("morph3d_") +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(isDilation) + std::to_string(SeLength);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::string options = generateOptionsString<T, isDilation, SeLength>();
        const char* ker_strs[] = {morph_cl};
        const int   ker_lens[] = {morph_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "morph3d");
        addKernelToCache(device, refName, entry);
    }

    auto morphOp = KernelFunctor< Buffer, KParam, Buffer, KParam, Buffer,
                                  cl::LocalSpaceArg, int >(*entry.ker);

    NDRange local(CUBE_X, CUBE_Y, CUBE_Z);

    int blk_x = divup(in.info.dims[0], CUBE_X);
    int blk_y = divup(in.info.dims[1], CUBE_Y);
    int blk_z = divup(in.info.dims[2], CUBE_Z);
    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * CUBE_X * in.info.dims[3], blk_y * CUBE_Y, blk_z * CUBE_Z);

    // copy mask/filter to constant memory
    cl_int se_size   = sizeof(T)*SeLength*SeLength*SeLength;
    cl::Buffer *mBuff = bufferAlloc(se_size);
    getQueue().enqueueCopyBuffer(*mask.data, *mBuff, 0, 0, se_size);

    // calculate shared memory size
    const int padding = (SeLength%2==0 ? (SeLength-1) : (2*(SeLength/2)));
    const int locLen  = CUBE_X+padding+1;
    const int locArea = locLen *(CUBE_Y+padding);
    const int locSize = locArea*(CUBE_Z+padding);

    morphOp(EnqueueArgs(getQueue(), global, local),
            *out.data, out.info, *in.data, in.info,
            *mBuff, cl::Local(locSize*sizeof(T)), blk_x);

    bufferFree(mBuff);
    CL_DEBUG_FINISH(getQueue());
}
}
}
