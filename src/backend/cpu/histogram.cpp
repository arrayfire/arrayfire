/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <histogram.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/histogram.hpp>

using af::dim4;

namespace cpu
{

template<typename inType, typename outType, bool isLinear>
Array<outType> histogram(const Array<inType> &in,
                         const unsigned &nbins,
                         const double &minval, const double &maxval)
{
    in.eval();

    const dim4 inDims  = in.dims();
    dim4 outDims       = dim4(nbins,1,inDims[2],inDims[3]);
    Array<outType> out = createValueArray<outType>(outDims, outType(0));
    out.eval();

    getQueue().enqueue(kernel::histogram<outType, inType, isLinear>,
            out, in, nbins, minval, maxval);

    return out;
}

#define INSTANTIATE(in_t,out_t)\
template Array<out_t> histogram<in_t, out_t, true>(const Array<in_t> &in, const unsigned &nbins, const double &minval, const double &maxval); \
template Array<out_t> histogram<in_t, out_t, false>(const Array<in_t> &in, const unsigned &nbins, const double &minval, const double &maxval);

INSTANTIATE(float , uint)
INSTANTIATE(double, uint)
INSTANTIATE(char  , uint)
INSTANTIATE(int   , uint)
INSTANTIATE(uint  , uint)
INSTANTIATE(uchar , uint)
INSTANTIATE(short , uint)
INSTANTIATE(ushort, uint)
INSTANTIATE(intl  , uint)
INSTANTIATE(uintl , uint)

}
