/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <histogram.hpp>
#include <kernel/histogram.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<uint> histogram(const Array<T> &in, const unsigned &nbins,
                      const double &minval, const double &maxval,
                      const bool isLinear) {
    const dim4 &inDims = in.dims();
    dim4 outDims       = dim4(nbins, 1, inDims[2], inDims[3]);
    Array<uint> out    = createValueArray<uint>(outDims, uint(0));
    if (isLinear) {
        getQueue().enqueue(kernel::histogram<T, true>, out, in, nbins, minval,
                           maxval);
    } else {
        getQueue().enqueue(kernel::histogram<T, false>, out, in, nbins, minval,
                           maxval);
    }
    return out;
}

#define INSTANTIATE(T)                                                    \
    template Array<uint> histogram<T>(const Array<T> &, const unsigned &, \
                                      const double &, const double &,     \
                                      const bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace arrayfire
