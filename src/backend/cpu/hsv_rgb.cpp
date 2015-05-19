/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <ArrayInfo.hpp>
#include <hsv_rgb.hpp>
#include <err_cpu.hpp>
#include <cmath>

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> hsv2rgb(const Array<T>& in)
{
    const dim4 dims    = in.dims();
    const dim4 strides = in.strides();
    Array<T> out       = createEmptyArray<T>(dims);
    dim_t obStride  = out.strides()[3];
    dim_t coff      = strides[2];
    dim_t bCount    = dims[3];

    for(dim_t b=0; b<bCount; ++b) {
        const T* src = in.get() + b * strides[3];
        T* dst       = out.get() + b * obStride;

        for(dim_t j=0; j<dims[1]; ++j) {
            dim_t jOff = j*strides[1];
            // j steps along 2nd dimension
            for(dim_t i=0; i<dims[0]; ++i) {
                // i steps along 1st dimension
                dim_t hIdx = i*strides[0] + jOff;
                dim_t sIdx = hIdx + coff;
                dim_t vIdx = sIdx + coff;

                T H = src[hIdx];
                T S = src[sIdx];
                T V = src[vIdx];

                T R, G, B;
                R = G = B = 0;

                int   m = (int)(H * 6);
                T f = H * 6 - m;
                T p = V * (1 - S);
                T q = V * (1 - f * S);
                T t = V * (1 - (1 - f) * S);

                switch (m % 6) {
                    case 0: R = V, G = t, B = p; break;
                    case 1: R = q, G = V, B = p; break;
                    case 2: R = p, G = V, B = t; break;
                    case 3: R = p, G = q, B = V; break;
                    case 4: R = t, G = p, B = V; break;
                    case 5: R = V, G = p, B = q; break;
                }

                dst[hIdx] = R;
                dst[sIdx] = G;
                dst[vIdx] = B;
            }
        }
    }

    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in)
{
    const dim4 dims    = in.dims();
    const dim4 strides = in.strides();
    Array<T> out       = createEmptyArray<T>(dims);
    dim4 oStrides      = out.strides();
    dim_t bCount    = dims[3];

    for(dim_t b=0; b<bCount; ++b) {
        const T* src = in.get() + b * strides[3];
        T* dst       = out.get() + b * oStrides[3];

        for(dim_t j=0; j<dims[1]; ++j) {
            // j steps along 2nd dimension
            dim_t oj = j * oStrides[1];
            dim_t ij = j * strides[1];

            for(dim_t i=0; i<dims[0]; ++i) {
                // i steps along 1st dimension
                dim_t oIdx0 = i * oStrides[0] + oj;
                dim_t oIdx1 = oIdx0 + oStrides[2];
                dim_t oIdx2 = oIdx1 + oStrides[2];

                dim_t iIdx0 = i * strides[0]  + ij;
                dim_t iIdx1 = iIdx0 + strides[2];
                dim_t iIdx2 = iIdx1 + strides[2];

                T R = src[iIdx0];
                T G = src[iIdx1];
                T B = src[iIdx2];
                T Cmax = std::max(std::max(R, G), B);
                T Cmin = std::min(std::min(R, G), B);
                T delta= Cmax-Cmin;

                T H = 0;

                if (Cmax!=Cmin) {
                    if (Cmax==R) H = (G-B)/delta + (G<B ? 6 : 0);
                    if (Cmax==G) H = (B-R)/delta + 2;
                    if (Cmax==B) H = (R-G)/delta + 4;
                    H = H / 6.0f;
                }

                dst[oIdx0] = H;
                dst[oIdx1] = (Cmax==0.0f ? 0 : delta/Cmax);
                dst[oIdx2] = Cmax;
            }
        }
    }

    return out;
}

#define INSTANTIATE(T)  \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in); \

INSTANTIATE(double)
INSTANTIATE(float )

}
