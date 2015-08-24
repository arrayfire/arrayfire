/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/convolve_separable.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <memory.hpp>

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

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, typename accType, int conv_dim, bool expand, int fLen>
void convolve2(Param out, const Param signal, const Param filter)
{
    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>   convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                const size_t C0_SIZE  = (THREADS_X+2*(fLen-1))* THREADS_Y;
                const size_t C1_SIZE  = (THREADS_Y+2*(fLen-1))* THREADS_X;

                size_t locSize = (conv_dim==0 ? C0_SIZE : C1_SIZE);

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<accType>::getName()
                            << " -D CONV_DIM="<< conv_dim
                            << " -D EXPAND="<< expand
                            << " -D FLEN="<< fLen
                            << " -D LOCAL_MEM_SIZE="<<locSize;
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, convolve_separable_cl, convolve_separable_cl_len, options.str());
                    convProgs[device]   = new Program(prog);
                    convKernels[device] = new Kernel(*convProgs[device], "convolve");
                });

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam, Buffer,
                                  int, int>(*convKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(out.info.dims[0], THREADS_X);
        int blk_y = divup(out.info.dims[1], THREADS_Y);

        NDRange global(blk_x*signal.info.dims[2]*THREADS_X,
                       blk_y*signal.info.dims[3]*THREADS_Y);

        cl::Buffer *mBuff = bufferAlloc(fLen*sizeof(accType));
        // FIX ME: if the filter array is strided, direct might cause issues
        getQueue().enqueueCopyBuffer(*filter.data, *mBuff, 0, 0, fLen*sizeof(accType));

        convOp(EnqueueArgs(getQueue(), global, local),
               *out.data, out.info, *signal.data, signal.info, *mBuff, blk_x, blk_y);

        bufferFree(mBuff);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename accT, dim_t cDim, bool expand>
void conv2Helper(Param out, const Param sig, const Param filt, dim_t f)
{
    switch(f) {
        case  2: kernel::convolve2<T, accT, cDim, expand,  2>(out, sig, filt); break;
        case  3: kernel::convolve2<T, accT, cDim, expand,  3>(out, sig, filt); break;
        case  4: kernel::convolve2<T, accT, cDim, expand,  4>(out, sig, filt); break;
        case  5: kernel::convolve2<T, accT, cDim, expand,  5>(out, sig, filt); break;
        case  6: kernel::convolve2<T, accT, cDim, expand,  6>(out, sig, filt); break;
        case  7: kernel::convolve2<T, accT, cDim, expand,  7>(out, sig, filt); break;
        case  8: kernel::convolve2<T, accT, cDim, expand,  8>(out, sig, filt); break;
        case  9: kernel::convolve2<T, accT, cDim, expand,  9>(out, sig, filt); break;
        case 10: kernel::convolve2<T, accT, cDim, expand, 10>(out, sig, filt); break;
        case 11: kernel::convolve2<T, accT, cDim, expand, 11>(out, sig, filt); break;
        case 12: kernel::convolve2<T, accT, cDim, expand, 12>(out, sig, filt); break;
        case 13: kernel::convolve2<T, accT, cDim, expand, 13>(out, sig, filt); break;
        case 14: kernel::convolve2<T, accT, cDim, expand, 14>(out, sig, filt); break;
        case 15: kernel::convolve2<T, accT, cDim, expand, 15>(out, sig, filt); break;
        case 16: kernel::convolve2<T, accT, cDim, expand, 16>(out, sig, filt); break;
        case 17: kernel::convolve2<T, accT, cDim, expand, 17>(out, sig, filt); break;
        case 18: kernel::convolve2<T, accT, cDim, expand, 18>(out, sig, filt); break;
        case 19: kernel::convolve2<T, accT, cDim, expand, 19>(out, sig, filt); break;
        case 20: kernel::convolve2<T, accT, cDim, expand, 20>(out, sig, filt); break;
        case 21: kernel::convolve2<T, accT, cDim, expand, 21>(out, sig, filt); break;
        case 22: kernel::convolve2<T, accT, cDim, expand, 22>(out, sig, filt); break;
        case 23: kernel::convolve2<T, accT, cDim, expand, 23>(out, sig, filt); break;
        case 24: kernel::convolve2<T, accT, cDim, expand, 24>(out, sig, filt); break;
        case 25: kernel::convolve2<T, accT, cDim, expand, 25>(out, sig, filt); break;
        case 26: kernel::convolve2<T, accT, cDim, expand, 26>(out, sig, filt); break;
        case 27: kernel::convolve2<T, accT, cDim, expand, 27>(out, sig, filt); break;
        case 28: kernel::convolve2<T, accT, cDim, expand, 28>(out, sig, filt); break;
        case 29: kernel::convolve2<T, accT, cDim, expand, 29>(out, sig, filt); break;
        case 30: kernel::convolve2<T, accT, cDim, expand, 30>(out, sig, filt); break;
        case 31: kernel::convolve2<T, accT, cDim, expand, 31>(out, sig, filt); break;
        default: OPENCL_NOT_SUPPORTED();
    }
}

#define INSTANTIATE(T, accT)  \
    template void conv2Helper<T, accT, 0, true >(Param out, const Param sig, const Param filt, dim_t f); \
    template void conv2Helper<T, accT, 1, true >(Param out, const Param sig, const Param filt, dim_t f); \
    template void conv2Helper<T, accT, 0, false>(Param out, const Param sig, const Param filt, dim_t f); \
    template void conv2Helper<T, accT, 1, false>(Param out, const Param sig, const Param filt, dim_t f);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

}
