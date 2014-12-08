/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/convolve.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
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

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const dim_type MAX_CONV1_FILTER_LEN = 129;
static const dim_type MAX_CONV2_FILTER_LEN = 17;
static const dim_type MAX_CONV3_FILTER_LEN = 5;

template<typename T, dim_type baseDim>
void prepareKernelArgs(NDRange &global, NDRange &local, size_t &loc_size, dim_type &blk_x,
                       ConvolveBatchKind kind, dim_type *oDims, const dim_type *sDims,
                       const dim_type *fDims)
{
    dim_type blk_y, blk_z;
    if (baseDim==1) {
        local = NDRange(THREADS, 1);
        blk_x = divup(oDims[0], THREADS);
        if (kind==MANY2ONE)
            global = NDRange(blk_x*THREADS*sDims[1], 1);
        else
            global = NDRange(blk_x*THREADS, 1);
        loc_size = (THREADS+2*(fDims[0]-1)) * sizeof(T);
    } else if (baseDim==2) {
        local = NDRange(THREADS_X, THREADS_Y);
        blk_x = divup(oDims[0], THREADS_X);
        blk_y = divup(oDims[1], THREADS_Y);
        if (kind==MANY2ONE)
            global = NDRange(blk_x*THREADS_X*sDims[2], blk_y*THREADS_Y);
        else
            global = NDRange(blk_x*THREADS_X, blk_y*THREADS_Y);
        loc_size = (THREADS_X+2*(fDims[0]-1))*(THREADS_Y+2*(fDims[1]-1)) * sizeof(T);
    } else if (baseDim==3) {
        local = NDRange(CUBE_X, CUBE_Y, CUBE_Z);
        blk_x = divup(oDims[0], CUBE_X);
        blk_y = divup(oDims[1], CUBE_Y);
        blk_z = divup(oDims[2], CUBE_Z);
        if (kind==MANY2ONE)
            global = NDRange(blk_x*CUBE_X*sDims[3], blk_y*CUBE_Y, blk_z*CUBE_Z);
        else
            global = NDRange(blk_x*CUBE_X, blk_y*CUBE_Y, blk_z*CUBE_Z);
        loc_size = (CUBE_X+2*(fDims[0]-1)) * (CUBE_Y+2*(fDims[1]-1)) *
                   (CUBE_Z+2*(fDims[2]-1)) * sizeof(T);
    }
}

template<typename T, typename accType, dim_type baseDim, bool expand>
void convolve_nd(Param out, const Param signal, const Param filter, ConvolveBatchKind kind)
{
    dim_type bCount   = 1ll;
    dim_type steps[3] = { 0ll, 0ll, 0ll };
    // [0] - output step, [1] - signal step, [2] - filter step
    if (kind==MANY2MANY) {
        steps[0] = out.info.strides[baseDim];
        steps[1] = signal.info.strides[baseDim];
        steps[2] = filter.info.strides[baseDim];
        bCount   = signal.info.dims[baseDim];
    } else if (kind==ONE2ALL) {
        steps[0] = out.info.strides[baseDim];
        steps[2] = filter.info.strides[baseDim];
        bCount   = filter.info.dims[baseDim];
    }

    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<T>::getName()
                            << " -D BASE_DIM="<< baseDim
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

        // prepare launch parameters
        NDRange global, local;
        size_t loc_size;
        dim_type blk_x;
        prepareKernelArgs<T, baseDim>(global, local, loc_size, blk_x, kind,
                out.info.dims, signal.info.dims, filter.info.dims);

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                  cl::LocalSpaceArg, Buffer, KParam,
                                  dim_type, dim_type, dim_type>(*convKernels[device]);

        cl_int se_size;
        switch(baseDim) {
            case 1: se_size = sizeof(T)*filter.info.dims[0]; break;
            case 2: se_size = sizeof(T)*filter.info.dims[0]*filter.info.dims[1]; break;
            case 3: se_size = sizeof(T)*filter.info.dims[0]*filter.info.dims[1]*filter.info.dims[2]; break;
        }

        cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, se_size);

        for (dim_type b=0; b<bCount; ++b) {
            // FIX ME: if the filter array is strided, direct copy might cause issues
            getQueue().enqueueCopyBuffer(*filter.data, mBuff, b*steps[2]*sizeof(T), 0, se_size);

            convOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *signal.data, signal.info, cl::Local(loc_size),
                    mBuff, filter.info, blk_x, b*steps[0], b*steps[1]);
        }
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
void convolve2(Param out, const Param signal, const Param filter)
{
    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>   convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<T>::getName()
                            << " -D CONV_DIM="<< conv_dim
                            << " -D EXPAND="<< expand
                            << " -D SEPARABLE_CONV";
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, convolve_cl, convolve_cl_len, options.str());
                    convProgs[device]   = new Program(prog);
                    convKernels[device] = new Kernel(*convProgs[device], "convolve");
                });

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam, cl::LocalSpaceArg,
                                  Buffer, dim_type, dim_type>(*convKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(out.info.dims[0], THREADS_X);
        dim_type blk_y = divup(out.info.dims[1], THREADS_Y);

        NDRange global(blk_x*signal.info.dims[2]*THREADS_X, blk_y*THREADS_Y);

        dim_type fLen = filter.info.dims[0];
        size_t loc_size = 0;
        if (conv_dim==0)
           loc_size = (THREADS_X+2*(fLen-1))*THREADS_Y * sizeof(T);
        else if(conv_dim==1)
           loc_size = (THREADS_Y+2*(fLen-1))*THREADS_X * sizeof(T);

        cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, fLen*sizeof(T));
        // FIX ME: if the filter array is strided, direct might cause issues
        getQueue().enqueueCopyBuffer(*filter.data, mBuff, 0, 0, fLen*sizeof(T));

        convOp(EnqueueArgs(getQueue(), global, local),
               *out.data, out.info, *signal.data, signal.info,
               cl::Local(loc_size), mBuff, fLen, blk_x);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
