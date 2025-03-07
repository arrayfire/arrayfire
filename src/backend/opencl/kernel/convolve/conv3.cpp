/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/convolve/conv_common.hpp>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, typename aT>
void conv3(conv_kparam_t& p, Param& out, const Param& sig, const Param& filt,
           const bool expand) {
    size_t se_size =
        filt.info.dims[0] * filt.info.dims[1] * filt.info.dims[2] * sizeof(aT);
    p.impulse = bufferAlloc(se_size);
    int f0Off = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];
        // FIXME: if the filter array is strided, direct copy of symbols
        // might cause issues
        getQueue().enqueueCopyBuffer(*filt.data, *p.impulse,
                                     (f0Off + f3Off) * sizeof(aT), 0, se_size);

        p.o[2] = (p.outHasNoOffset ? 0 : b3);
        p.s[2] = (p.inHasNoOffset ? 0 : b3);

        convNHelper<T, aT>(p, out, sig, filt, 3, expand);
    }
}

#define INSTANTIATE(T, accT)                                           \
    template void conv3<T, accT>(conv_kparam_t&, Param&, const Param&, \
                                 const Param&, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
