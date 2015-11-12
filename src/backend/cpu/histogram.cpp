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

using af::dim4;

namespace cpu
{

template<typename inType, typename outType, bool isLinear>
Array<outType> histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval)
{
    float step = (maxval - minval)/(float)nbins;

    const dim4 inDims  = in.dims();
    dim4 iStrides      = in.strides();
    dim4 outDims       = dim4(nbins,1,inDims[2],inDims[3]);
    Array<outType> out = createValueArray<outType>(outDims, outType(0));
    dim4 oStrides      = out.strides();
    dim_t nElems    = inDims[0]*inDims[1];

    outType *outData    = out.get();
    const inType* inData= in.get();

    for(dim_t b3 = 0; b3 < outDims[3]; b3++) {
        for(dim_t b2 = 0; b2 < outDims[2]; b2++) {
            for(dim_t i=0; i<nElems; i++) {
                int idx = isLinear ? i : ((i % inDims[0]) + (i / inDims[0])*iStrides[1]);
                int bin = (int)((inData[idx] - minval) / step);
                bin = std::max(bin, 0);
                bin = std::min(bin, (int)(nbins - 1));
                outData[bin]++;
            }
            inData  += iStrides[2];
            outData += oStrides[2];
        }
    }

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
