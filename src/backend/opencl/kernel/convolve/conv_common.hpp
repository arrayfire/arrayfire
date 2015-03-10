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

#include <convolve_common.hpp>
#include <kernel_headers/convolve.hpp>

#include <map>
#include <mutex>
#include <string>
#include <Param.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <dispatch.hpp>
#include <platform.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{

namespace kernel
{

static const dim_type THREADS   = 256;

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

static const dim_type CUBE_X    =  8;
static const dim_type CUBE_Y    =  8;
static const dim_type CUBE_Z    =  4;

struct conv_kparam_t {
    NDRange             global;
    NDRange              local;
    size_t            loc_size;
    ConvolveBatchKind     kind;
    dim_type              nBBS;
    dim_type            bCount;
    dim_type          steps[3];
};

template<typename T, dim_type baseDim>
void prepareKernelArgs(conv_kparam_t& param, ConvolveBatchKind kind,
                       dim_type *oDims, const dim_type *sDims, const dim_type *fDims,
                       dim_type *oStrides, const dim_type *sStrides, const dim_type *fStrides)
{
    param.bCount = 1ll;
    param.kind   = kind;

    for(dim_type i=0; i<3; ++i)
        param.steps[i] = 0ll;

    // [0] - output step, [1] - signal step, [2] - filter step
    if (kind==MANY2MANY) {
        param.steps[0] = oStrides[baseDim];
        param.steps[1] = sStrides[baseDim];
        param.steps[2] = fStrides[baseDim];
        param.bCount   = sDims[baseDim];
    } else if (kind==ONE2ALL) {
        param.steps[0] = oStrides[baseDim];
        param.steps[2] = fStrides[baseDim];
        param.bCount   = fDims[baseDim];
    }

    dim_type blk_y, blk_z;
    if (baseDim==1) {
        param.local = NDRange(THREADS, 1);
        param.nBBS = divup(oDims[0], THREADS);

        if (kind==MANY2ONE)
            param.global = NDRange(param.nBBS*THREADS*sDims[1], 1);
        else
            param.global = NDRange(param.nBBS*THREADS, 1);

        param.loc_size = (THREADS+2*(fDims[0]-1)) * sizeof(T);
    } else if (baseDim==2) {
        param.local = NDRange(THREADS_X, THREADS_Y);
        param.nBBS = divup(oDims[0], THREADS_X);
        blk_y = divup(oDims[1], THREADS_Y);

        if (kind==MANY2ONE)
            param.global = NDRange(param.nBBS*THREADS_X*sDims[2], blk_y*THREADS_Y);
        else
            param.global = NDRange(param.nBBS*THREADS_X, blk_y*THREADS_Y);

        param.loc_size = (THREADS_X+2*(fDims[0]-1))*(THREADS_Y+2*(fDims[1]-1)) * sizeof(T);
    } else if (baseDim==3) {
        param.local = NDRange(CUBE_X, CUBE_Y, CUBE_Z);
        param.nBBS = divup(oDims[0], CUBE_X);
        blk_y = divup(oDims[1], CUBE_Y);
        blk_z = divup(oDims[2], CUBE_Z);

        if (kind==MANY2ONE)
            param.global = NDRange(param.nBBS*CUBE_X*sDims[3], blk_y*CUBE_Y, blk_z*CUBE_Z);
        else
            param.global = NDRange(param.nBBS*CUBE_X, blk_y*CUBE_Y, blk_z*CUBE_Z);

        param.loc_size = (CUBE_X+2*(fDims[0]-1)) * (CUBE_Y+2*(fDims[1]-1)) *
                         (CUBE_Z+2*(fDims[2]-1)) * sizeof(T);
    }
}

template<typename T, typename accType, dim_type bDim, bool expand>
void convNHelper(const conv_kparam_t& param, Param& out, const Param& signal, const Param& filter)
{
    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<accType>::getName()
                            << " -D BASE_DIM="<< bDim
                            << " -D EXPAND=" << expand;
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, convolve_cl, convolve_cl_len, options.str());
                    convProgs[device]   = new Program(prog);
                    convKernels[device] = new Kernel(*convProgs[device], "convolve");
                });

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                  cl::LocalSpaceArg, Buffer, KParam,
                                  dim_type, dim_type, dim_type>(*convKernels[device]);

        cl_int se_size;
        switch(bDim) {
            case 1: se_size = sizeof(accType)*filter.info.dims[0]; break;
            case 3: se_size = sizeof(accType)*filter.info.dims[0]*filter.info.dims[1]*filter.info.dims[2]; break;
        }

        cl::Buffer *mBuff = bufferAlloc(se_size);

        for (dim_type b=0; b<param.bCount; ++b) {
            // FIX ME: if the filter array is strided, direct copy might cause issues
            getQueue().enqueueCopyBuffer(*filter.data, *mBuff, b*param.steps[2]*sizeof(accType), 0, se_size);

            convOp(EnqueueArgs(getQueue(), param.global, param.local),
                    *out.data, out.info, *signal.data, signal.info, cl::Local(param.loc_size),
                    *mBuff, filter.info, param.nBBS, b*param.steps[0], b*param.steps[1]);
        }

        bufferFree(mBuff);

    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename accType, bool expand>
void conv1(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);

template<typename T, typename accType, bool expand>
void conv2(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);

template<typename T, typename accType, bool expand>
void conv3(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt);

}

}
