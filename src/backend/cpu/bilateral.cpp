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

static inline dim_t clamp(int a, dim_t mn, dim_t mx)
{
    return (a < (int)mn ? mn : (a > (int)mx ? mx : a));
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

    outType *outData    = out.get();
    const inType * inData = in.get();

    // clamp spatical and chromatic sigma's
    float space_          = std::min(11.5f, std::max(s_sigma, 0.f));
    float color_          = std::max(c_sigma, 0.f);
    const dim_t radius = std::max((dim_t)(space_ * 1.5f), (dim_t)1);
    const float svar      = space_*space_;
    const float cvar      = color_*color_;

    for(dim_t b3=0; b3<dims[3]; ++b3) {
        // b3 for loop handles following batch configurations
        //  - gfor
        //  - input based batch
        //      - when input is 4d array for color images
        for(dim_t b2=0; b2<dims[2]; ++b2) {
            // b2 for loop handles following batch configurations
            //  - channels
            //  - input based batch
            //      - when input is 3d array for grayscale images
            for(dim_t j=0; j<dims[1]; ++j) {
                // j steps along 2nd dimension
                for(dim_t i=0; i<dims[0]; ++i) {
                    // i steps along 1st dimension
                    outType norm = 0.0;
                    outType res  = 0.0;
                    const outType center = (outType)inData[getIdx(istrides, i, j)];
                    for(dim_t wj=-radius; wj<=radius; ++wj) {
                        // clamps offsets
                        dim_t tj = clamp(j+wj, 0, dims[1]-1);
                        for(dim_t wi=-radius; wi<=radius; ++wi) {
                            // clamps offsets
                            dim_t ti = clamp(i+wi, 0, dims[0]-1);
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
