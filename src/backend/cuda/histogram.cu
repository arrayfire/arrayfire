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

using af::dim4;

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
    cfloat* h_minmax = new cfloat[dims[2]];

    for(dim_type k=0; k<dims[2]; ++k) {
        h_minmax[k].x = minval;
        h_minmax[k].y = maxval;
    }

    dim4 minmax_dims(dims[2]*2);
    Array<cfloat> minmax = createHostDataArray<cfloat>(minmax_dims, h_minmax);

    // cleanup the host memory used
    delete[] h_minmax;

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
