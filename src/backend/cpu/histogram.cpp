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

template<typename inType, typename outType>
Array<outType> histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval)
{
    const dim4 inDims   = in.dims();
    dim4 outDims        = dim4(nbins,1,inDims[2],inDims[3]);

    // create an array with first two dimensions swapped
    Array<outType> out = createEmptyArray<outType>(outDims);

    // get data pointers for input and output Arrays
    outType *outData    = out.get();
    const inType* inData= in.get();

    dim_type batchCount = inDims[2] * inDims[3];
    dim_type batchStride= in.strides()[2];
    dim_type numElements= inDims[0]*inDims[1];

    // set all bin elements to zero
    outType *temp = outData;
    for(int batchId = 0; batchId < batchCount; batchId++) {
        for(int i=0; i < (int)nbins; i++)
            temp[i] = 0;
        temp += nbins;
    }

    float step = (maxval - minval)/(float)nbins;

    for(dim_type batchId = 0; batchId < batchCount; batchId++) {
        for(dim_type i=0; i<numElements; i++) {
            int bin = (int)((inData[i] - minval) / step);
            bin = std::max(bin, 0);
            bin = std::min(bin, (int)(nbins - 1));
            outData[bin]++;
        }
        inData  += batchStride;
        outData += nbins;
    }

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
