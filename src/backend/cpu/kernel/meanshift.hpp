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
#include <vector>
#include <utility.hpp>
#include <type_traits>

namespace cpu
{
namespace kernel
{
template<typename T, bool IsColor>
void meanShift(Param<T> out, CParam<T> in, const float s_sigma,
               const float c_sigma, const unsigned iter)
{
    typedef typename std::conditional< std::is_same<T, double>::value, double, float >::type AccType;

    const af::dim4 dims     = in.dims();
    const af::dim4 istrides = in.strides();
    const af::dim4 ostrides = out.strides();
    const unsigned bCount   = (IsColor ? 1 : dims[2]);
    const unsigned channels = (IsColor ? dims[2] : 1);
    float space_            = std::min(11.5f, s_sigma);
    const dim_t radius      = std::max((int)(space_ * 1.5f), 1);
    const AccType cvar      = c_sigma*c_sigma;

    for (dim_t b3=0; b3<dims[3]; ++b3) {
        for (unsigned b2=0; b2<bCount; ++b2) {

            T *      outData = out.get() + b2 * ostrides[2] + b3 * ostrides[3];
            const T * inData = in.get()  + b2 * istrides[2] + b3 * istrides[3];

            for (dim_t j=0; j<dims[1]; ++j) {

                dim_t j_in_off  = j*istrides[1];
                dim_t j_out_off = j*ostrides[1];

                for (dim_t i=0; i<dims[0]; ++i) {

                    dim_t i_in_off  = i*istrides[0];
                    dim_t i_out_off = i*ostrides[0];

                    std::vector<T> centers(channels, 0);

                    for (unsigned ch=0; ch<channels; ++ch)
                        centers[ch] = inData[j_in_off + i_in_off + ch*istrides[2]];

                    int cj = j;
                    int ci = i;

                    // scope of meanshift iterations begin
                    for (unsigned it=0; it<iter; ++it) {

                        int ocj   = cj;
                        int oci   = ci;
                        unsigned count = 0;
                        int shift_y = 0;
                        int shift_x = 0;

                        std::vector<AccType> means(channels, 0);

                        // Windowing operation
                        for (dim_t wj=-radius; wj<=radius; ++wj) {

                            int hit_count = 0;
                            dim_t tj = cj + wj;
                            if (tj<0 || tj>dims[1]-1) break;

                            dim_t tjstride = tj*istrides[1];

                            for (dim_t wi=-radius; wi<=radius; ++wi) {

                                dim_t ti = ci + wi;
                                if (ti<0 || ti>dims[0]-1) break;

                                dim_t tistride = ti*istrides[0];

                                std::vector<T> tmpclrs(channels, 0);

                                AccType norm = 0;
                                for (unsigned ch=0; ch<channels; ++ch) {
                                    tmpclrs[ch] = inData[ tistride + tjstride + ch*istrides[2] ];
                                    AccType diff = static_cast<AccType>(centers[ch]) -
                                                   static_cast<AccType>(tmpclrs[ch]);
                                    norm += (diff * diff);
                                }

                                if (norm <= cvar) {
                                    for(unsigned ch=0; ch<channels; ++ch)
                                        means[ch] += static_cast<AccType>(tmpclrs[ch]);

                                    shift_x += ti;
                                    ++hit_count;
                                }
                            }
                            count   += hit_count;
                            shift_y += tj*hit_count;
                        }

                        if (count==0) break;

                        const AccType fcount = 1/static_cast<AccType>(count);

                        cj = static_cast<int>(std::trunc(shift_y*fcount));
                        ci = static_cast<int>(std::trunc(shift_x*fcount));

                        for (unsigned ch=0; ch<channels; ++ch)
                            means[ch] = std::trunc(means[ch]*fcount);

                        AccType norm = 0;
                        for (unsigned ch=0; ch<channels; ++ch) {
                            AccType diff = means[ch] - static_cast<AccType>(centers[ch]);
                            norm += (diff*diff);
                        }

                        //stop the process if mean converged or within given tolerance range
                        bool stop = (cj==ocj && oci==ci) || ((abs(ocj-cj) + abs(oci-ci) + norm) <= 1);

                        for (unsigned ch=0; ch<channels; ++ch)
                            centers[ch] = static_cast<T>(means[ch]);

                        if (stop) break;
                    } // scope of meanshift iterations end

                    for (dim_t ch=0; ch<channels; ++ch)
                        outData[j_out_off + i_out_off + ch*ostrides[2]] = centers[ch];
                }
            }
        }
    }
}
}
}
