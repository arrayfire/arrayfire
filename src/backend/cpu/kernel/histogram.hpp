/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace cpu
{
namespace kernel
{

template<typename OutT, typename InT, bool IsLinear>
void histogram(Param<OutT> out, CParam<InT> in,
               unsigned const nbins, double const minval, double const maxval)
{
    dim4 const outDims   = out.dims();
    float const step     = (maxval - minval)/(float)nbins;
    dim4 const inDims    = in.dims();
    dim4 const iStrides  = in.strides();
    dim4 const oStrides  = out.strides();
    dim_t const nElems   = inDims[0]*inDims[1];


    for(dim_t b3 = 0; b3 < outDims[3]; b3++) {
        OutT *outData    = out.get() + b3 * oStrides[3];
        const InT* inData= in.get() + b3 * iStrides[3];
        for(dim_t b2 = 0; b2 < outDims[2]; b2++) {
            for(dim_t i=0; i<nElems; i++) {
                int idx = IsLinear ? i : ((i % inDims[0]) + (i / inDims[0])*iStrides[1]);
                int bin = (int)((inData[idx] - minval) / step);
                bin = std::max(bin, 0);
                bin = std::min(bin, (int)(nbins - 1));
                outData[bin]++;
            }
            inData  += iStrides[2];
            outData += oStrides[2];
        }
    }
}

}
}
