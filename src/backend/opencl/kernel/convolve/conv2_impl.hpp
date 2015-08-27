/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/convolve/conv_common.hpp>
#include <cache.hpp>

namespace opencl
{

namespace kernel
{

template<typename T, typename aT, bool expand>
void conv2Helper(const conv_kparam_t& param, Param out, const Param signal, const Param filter)
{
    try {
        int f0 = filter.info.dims[0];
        int f1 = filter.info.dims[1];

        std::string ref_name =
            std::string("conv2_") +
            std::string(dtype_traits<T>::getName()) +
            std::string("_") +
            std::string(dtype_traits<aT>::getName()) +
            std::string("_") +
            std::to_string(expand) +
            std::string("_") +
            std::to_string(f0) +
            std::string("_") +
            std::to_string(f1);

        int device = getActiveDeviceId();
        kc_t::iterator idx = kernelCaches[device].find(ref_name);

        kc_entry_t entry;
        if (idx == kernelCaches[device].end()) {
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
            entry.prog   = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "convolve");

            kernelCaches[device][ref_name] = entry;
        } else {
            entry = idx->second;
        }

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                  Buffer, KParam, int, int,
                                  int, int,
                                  int, int
                                 >(*entry.ker);

        convOp(EnqueueArgs(getQueue(), param.global, param.local),
                *out.data, out.info, *signal.data, signal.info,
                *param.impulse, filter.info, param.nBBS0, param.nBBS1,
                param.o[1], param.o[2], param.s[1], param.s[2]);

    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
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
