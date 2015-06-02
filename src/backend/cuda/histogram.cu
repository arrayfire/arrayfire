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
#include <kernel/histogram.hpp>
#include <err_cuda.hpp>
#include <vector>

using af::dim4;
using std::vector;

namespace cuda
{

template<typename inType, typename outType>
Array<outType> histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval)
{

    ARG_ASSERT(1, (nbins<=kernel::MAX_BINS));

    const dim4 dims     = in.dims();
    dim4 outDims        = dim4(nbins, 1, dims[2], dims[3]);
    Array<outType> out  = createValueArray<outType>(outDims, outType(0));

    // create an array to hold min and max values for
    // batch operation handling, this will reduce
    // number of concurrent reads to one single memory location
    dim_t mmNElems= dims[2] * dims[3];
    cfloat init;
    init.x = minval;
    init.y = maxval;
    vector<cfloat> h_minmax(mmNElems, init);

    dim4 minmax_dims(mmNElems*2);
    Array<cfloat> minmax = createHostDataArray<cfloat>(minmax_dims, &h_minmax.front());

    kernel::histogram<inType, outType>(out, in, minmax.get(), nbins);

    return out;
}

#define INSTANTIATE(in_t,out_t)\
template Array<out_t> histogram(const Array<in_t> &in, const unsigned &nbins, const double &minval, const double &maxval);

INSTANTIATE(float , uint)
INSTANTIATE(double, uint)
INSTANTIATE(char  , uint)
INSTANTIATE(int   , uint)
INSTANTIATE(uint  , uint)
INSTANTIATE(uchar , uint)

}
