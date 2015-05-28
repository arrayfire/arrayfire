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

template<typename T, typename aT, bool expand>
void conv1(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt)
{
    size_t se_size = filt.info.dims[0] * sizeof(aT);
    p.impulse = bufferAlloc(se_size);
    int f0Off = filt.info.offset;

    for (int b3=0; b3<filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2=0; b2<filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            for (int b1=0; b1<filt.info.dims[1]; ++b1) {
                int f1Off = b1 * filt.info.strides[1];

                // FIXME: if the filter array is strided, direct copy of symbols
                // might cause issues
                getQueue().enqueueCopyBuffer(*filt.data, *p.impulse,
                                            (f0Off+f1Off+f2Off+f3Off)*sizeof(aT)
                                             , 0, se_size);

                p.o[0] = (p.outHasNoOffset ? 0 : b1);
                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[0] = (p.inHasNoOffset ? 0 : b1);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                convNHelper<T, aT, 1, expand>(p, out, sig, filt);
            }
        }
    }
}

#define INSTANTIATE(T, accT)  \
    template void conv1<T, accT, true >(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \
    template void conv1<T, accT, false>(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt); \

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
