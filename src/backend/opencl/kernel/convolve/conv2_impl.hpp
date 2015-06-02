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

template<typename T, typename aT, bool expand, int f0, int f1>
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
                            << " -D accType="<< dtype_traits<aT>::getName()
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
                                  Buffer, KParam, int, int,
                                  int, int,
                                  int, int
                                 >(*convKernels[device]);

        convOp(EnqueueArgs(getQueue(), param.global, param.local),
                *out.data, out.info, *signal.data, signal.info,
                *param.impulse, filter.info, param.nBBS0, param.nBBS1,
                param.o[1], param.o[2], param.s[1], param.s[2]);

    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename aT, bool expand, int f>
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
void conv2Helper(const conv_kparam_t& p, Param& out, const Param& sig, const Param& filt)
{
    int f0 = filt.info.dims[0];
    int f1 = filt.info.dims[1];
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
                             case 12: conv2Helper<T, aT, expand, 12, 12>(p, out, sig, filt); break;
                             case 13: conv2Helper<T, aT, expand, 13, 13>(p, out, sig, filt); break;
                             case 14: conv2Helper<T, aT, expand, 14, 14>(p, out, sig, filt); break;
                             case 15: conv2Helper<T, aT, expand, 15, 15>(p, out, sig, filt); break;
                             case 16: conv2Helper<T, aT, expand, 16, 16>(p, out, sig, filt); break;
                             case 17: conv2Helper<T, aT, expand, 17, 17>(p, out, sig, filt); break;
                             default: OPENCL_NOT_SUPPORTED();
                         }
                     } else
                         OPENCL_NOT_SUPPORTED();
                 } break;
    }
}

template<typename T, typename aT, bool expand>
void conv2(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt)
{
    size_t se_size = filt.info.dims[0] * filt.info.dims[1] * sizeof(aT);
    p.impulse = bufferAlloc(se_size);
    int f0Off = filt.info.offset;

    for (int b3=0; b3<filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2=0; b2<filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            // FIXME: if the filter array is strided, direct copy of symbols
            // might cause issues
            getQueue().enqueueCopyBuffer(*filt.data, *p.impulse,
                                         (f2Off+f3Off+f0Off)*sizeof(aT),
                                         0, se_size);

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<T, aT, expand>(p, out, sig, filt);
        }
    }
}

#define INSTANTIATE(T, accT)  \
    template void conv2<T, accT, true >(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \
    template void conv2<T, accT, false>(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \

}

}
