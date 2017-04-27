/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <vector>
#include <utility.hpp>

namespace cpu
{
namespace kernel
{

template<typename T, bool IsColor>
void meanShift(Array<T> out, const Array<T> in, const float s_sigma,
               const float c_sigma, const unsigned iter)
{
    const af::dim4 dims     = in.dims();
    const af::dim4 istrides = in.strides();
    const af::dim4 ostrides = out.strides();

    const dim_t bCount   = (IsColor ? 1 : dims[2]);
    const dim_t channels = (IsColor ? dims[2] : 1);

    // clamp spatical and chromatic sigma's
    float space_          = std::min(11.5f, s_sigma);
    const dim_t radius = std::max((int)(space_ * 1.5f), 1);
    const float cvar      = c_sigma*c_sigma;

    std::vector<float> means(channels);
    std::vector<float> centers(channels);
    std::vector<float> tmpclrs(channels);


    for(dim_t b3=0; b3<dims[3]; ++b3) {
        T *outData       = out.get() + b3 * ostrides[3];
        const T * inData = in.get() + b3 * istrides[3];

        for(dim_t b2=0; b2<bCount; ++b2) {

            for(dim_t j=0; j<dims[1]; ++j) {

                dim_t j_in_off  = j*istrides[1];
                dim_t j_out_off = j*ostrides[1];

                for(dim_t i=0; i<dims[0]; ++i) {

                    dim_t i_in_off  = i*istrides[0];
                    dim_t i_out_off = i*ostrides[0];

                    // clear means and centers for this pixel
                    for(dim_t ch=0; ch<channels; ++ch) {
                        means[ch] = 0.0f;
                        // the expression ch*istrides[2] will only effect when ch>1
                        // i.e for color images where batch is along fourth dimension
                        centers[ch] = inData[j_in_off + i_in_off + ch*istrides[2]];
                    }

                    // scope of meanshift iterationd begin
                    for(unsigned it=0; it<iter; ++it) {

                        int count   = 0;
                        int shift_x = 0;
                        int shift_y = 0;

                        for(dim_t wj=-radius; wj<=radius; ++wj) {

                            int hit_count = 0;

                            for(dim_t wi=-radius; wi<=radius; ++wi) {

                                dim_t tj = j + wj;
                                dim_t ti = i + wi;

                                // clamps offsets
                                tj = clamp(tj, 0ll, dims[1]-1);
                                ti = clamp(ti, 0ll, dims[0]-1);

                                // proceed
                                float norm = 0.0f;
                                for(dim_t ch=0; ch<channels; ++ch) {
                                    tmpclrs[ch] = inData[ tj*istrides[1] + ti*istrides[0] + ch*istrides[2]];
                                    norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                                }

                                if (norm<= cvar) {
                                    for(dim_t ch=0; ch<channels; ++ch)
                                        means[ch] += tmpclrs[ch];
                                    shift_x += wi;
                                    ++hit_count;
                                }

                            }
                            count+= hit_count;
                            shift_y += wj*hit_count;
                        }

                        if (count==0) { break; }

                        const float fcount = 1.f/count;
                        const int mean_x = (int)(shift_x*fcount+0.5f);
                        const int mean_y = (int)(shift_y*fcount+0.5f);
                        for(dim_t ch=0; ch<channels; ++ch)
                            means[ch] *= fcount;

                        float norm = 0.f;
                        for(dim_t ch=0; ch<channels; ++ch)
                            norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));
                        bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
                        shift_x = mean_x;
                        shift_y = mean_y;
                        for(dim_t ch=0; ch<channels; ++ch)
                            centers[ch] = means[ch];
                        if (stop) { break; }
                    } // scope of meanshift iterations end

                    for(dim_t ch=0; ch<channels; ++ch)
                        outData[j_out_off + i_out_off + ch*ostrides[2]] = centers[ch];

                }
            }
            outData += ostrides[2];
            inData  += istrides[2];
        }
    }
}

}
}
