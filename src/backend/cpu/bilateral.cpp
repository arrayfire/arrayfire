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
#include <bilateral.hpp>
#include <cmath>
#include <algorithm>

using af::dim4;

namespace cpu
{

static inline dim_type clamp(dim_type a, dim_type mn, dim_type mx)
{
    return (a<mn ? mn : (a>mx ? mx : a));
}

static inline unsigned getIdx(const dim4 &strides,
        int i, int j = 0, int k = 0, int l = 0)
{
    return (l * strides[3] +
            k * strides[2] +
            j * strides[1] +
            i * strides[0]);
}

template<typename inType, typename outType, bool isColor>
Array<outType> bilateral(const Array<inType> &in, const float &s_sigma, const float &c_sigma)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();

    Array<outType> out = createEmptyArray<outType>(dims);
    const dim4 ostrides = out.strides();

    dim_type bCount     = dims[2];
    if (isColor) bCount*= dims[3];

    outType *outData    = out.get();
    const inType * inData = in.get();

    // clamp spatical and chromatic sigma's
    float space_          = std::min(11.5f, std::max(s_sigma, 0.f));
    float color_          = std::max(c_sigma, 0.f);
    const dim_type radius = std::max((dim_type)(space_ * 1.5f), (dim_type)1);
    const float svar      = space_*space_;
    const float cvar      = color_*color_;

    for(dim_type batchId=0; batchId<bCount; ++batchId) {
        // channels or batch for gray and channel are handled by outer loop
        for(dim_type j=0; j<dims[1]; ++j) {
            // j steps along 2nd dimension
            for(dim_type i=0; i<dims[0]; ++i) {
                // i steps along 1st dimension
                outType norm = 0.0;
                outType res  = 0.0;
                const outType center = (outType)inData[getIdx(istrides, i, j)];
                for(dim_type wj=-radius; wj<=radius; ++wj) {
                    // clamps offsets
                    dim_type tj = clamp(j+wj, 0, dims[1]-1);
                    for(dim_type wi=-radius; wi<=radius; ++wi) {
                        // clamps offsets
                        dim_type ti = clamp(i+wi, 0, dims[0]-1);
                        // proceed
                        const outType val= (outType)inData[getIdx(istrides, ti, tj)];
                        const outType gauss_space = (wi*wi+wj*wj)/(-2.0*svar);
                        const outType gauss_range = ((center-val)*(center-val))/(-2.0*cvar);
                        const outType weight = std::exp(gauss_space+gauss_range);
                        norm += weight;
                        res += val*weight;
                    }
                } // filter loop ends here

                outData[getIdx(ostrides, i, j)] = res/norm;
            } //1st dimension loop ends here
        } //2nd dimension loop ends here
        outData += ostrides[2];
        inData  += istrides[2];
    }

    return out;
}

#define INSTANTIATE(inT, outT)\
template Array<outT> bilateral<inT, outT,true >(const Array<inT> &in, const float &s_sigma, const float &c_sigma);\
template Array<outT> bilateral<inT, outT,false>(const Array<inT> &in, const float &s_sigma, const float &c_sigma);

INSTANTIATE(double, double)
INSTANTIATE(float ,  float)
INSTANTIATE(char  ,  float)
INSTANTIATE(int   ,  float)
INSTANTIATE(uint  ,  float)
INSTANTIATE(uchar ,  float)

}
