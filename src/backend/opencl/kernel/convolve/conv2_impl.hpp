/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/convolve/conv_common.hpp>

namespace opencl
{

namespace kernel
{

template<typename T, typename aT, bool expand, dim_type f0, dim_type f1>
void conv2Helper(const conv_kparam_t& param, Param out, const Param signal, const Param filter)
{
    try {
        static std::once_flag  compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> convProgs;
        static std::map<int, Kernel*>  convKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    size_t LOC_SIZE = (THREADS_X+2*(f0-1))*(THREADS_Y+2*(f1-1));

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D accType="<< dtype_traits<T>::getName()
                            << " -D BASE_DIM="<< 2 /* hard constant specific to this convolution type */
                            << " -D FLEN0=" << f0
                            << " -D FLEN1=" << f1
                            << " -D EXPAND="<< expand
                            << " -D C_SIZE="<< LOC_SIZE;
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
                                  Buffer, KParam, dim_type, dim_type, dim_type>(*convKernels[device]);

        cl_int se_size    = sizeof(T)*filter.info.dims[0]*filter.info.dims[1];
        cl::Buffer *mBuff = bufferAlloc(se_size);

        for (dim_type b=0; b<param.bCount; ++b) {
            // FIX ME: if the filter array is strided, direct copy might cause issues
            getQueue().enqueueCopyBuffer(*filter.data, *mBuff, b*param.steps[2]*sizeof(T), 0, se_size);

            convOp(EnqueueArgs(getQueue(), param.global, param.local),
                    *out.data, out.info, *signal.data, signal.info,
                    *mBuff, filter.info, param.nBBS, b*param.steps[0], b*param.steps[1]);
        }
        bufferFree(mBuff);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename aT, bool expand, dim_type f>
void conv2Helper(const conv_kparam_t& p, Param out, const Param sig, const Param filt)
{
    switch(filt.info.dims[1]) {
        case  1: conv2Helper<T, aT, expand, f,  1>(p, out, sig, filt); break;
        case  2: conv2Helper<T, aT, expand, f,  2>(p, out, sig, filt); break;
        case  3: conv2Helper<T, aT, expand, f,  3>(p, out, sig, filt); break;
        case  4: conv2Helper<T, aT, expand, f,  4>(p, out, sig, filt); break;
        case  5: conv2Helper<T, aT, expand, f,  5>(p, out, sig, filt); break;
        default: OPENCL_NOT_SUPPORTED();
    }
}

template<typename T, typename aT, bool expand>
void conv2(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt)
{
    dim_type f0 = filt.info.dims[0];
    dim_type f1 = filt.info.dims[1];
    switch(f0) {
        case  1: conv2Helper<T, aT, expand,  1>(p, out, sig, filt); break;
        case  2: conv2Helper<T, aT, expand,  2>(p, out, sig, filt); break;
        case  3: conv2Helper<T, aT, expand,  3>(p, out, sig, filt); break;
        case  4: conv2Helper<T, aT, expand,  4>(p, out, sig, filt); break;
        case  5: conv2Helper<T, aT, expand,  5>(p, out, sig, filt); break;
        default: {
                     if (f0==f1) {
                         switch(f1) {
                             case  6: conv2Helper<T, aT, expand,  6,  6>(p, out, sig, filt); break;
                             case  7: conv2Helper<T, aT, expand,  7,  7>(p, out, sig, filt); break;
                             case  8: conv2Helper<T, aT, expand,  8,  8>(p, out, sig, filt); break;
                             case  9: conv2Helper<T, aT, expand,  9,  9>(p, out, sig, filt); break;
                             case 10: conv2Helper<T, aT, expand, 10, 10>(p, out, sig, filt); break;
                             case 11: conv2Helper<T, aT, expand, 11, 11>(p, out, sig, filt); break;
                             default: OPENCL_NOT_SUPPORTED();
                         }
                     } else
                         OPENCL_NOT_SUPPORTED();
                 } break;
    }
}

#define INSTANTIATE(T, accT)  \
    template void conv2<T, accT, true >(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \
    template void conv2<T, accT, false>(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \

}

}
