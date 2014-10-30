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
#include <meanshift.hpp>
#include <cmath>
#include <algorithm>
#include <err_cpu.hpp>

using af::dim4;
using std::vector;

namespace cpu
{

inline dim_type clamp(dim_type a, dim_type mn, dim_type mx)
{
    return (a<mn ? mn : (a>mx ? mx : a));
}

template<typename T, bool is_color>
Array<T> * meanshift(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();

    Array<T>* out       = createEmptyArray<T>(dims);
    const dim4 ostrides = out->strides();

    const dim_type bIndex   = (is_color ? 3ll : 2ll);
    const dim_type bCount   = dims[bIndex];
    const dim_type channels = (is_color ? dims[2] : 1ll);

    // clamp spatical and chromatic sigma's
    float space_          = std::min(11.5f, s_sigma);
    const dim_type radius = std::max((dim_type)(space_ * 1.5f), 1ll);
    const float cvar      = c_sigma*c_sigma;

    vector<float> means(channels);
    vector<float> centers(channels);
    vector<float> tmpclrs(channels);

    for(dim_type batchId=0; batchId<bCount; ++batchId) {

        T *outData       = out->get() + batchId*ostrides[bIndex];
        const T * inData = in.get()   + batchId*istrides[bIndex];

        for(dim_type j=0; j<dims[1]; ++j) {

            dim_type j_in_off  = j*istrides[1];
            dim_type j_out_off = j*ostrides[1];

            for(dim_type i=0; i<dims[0]; ++i) {

                dim_type i_in_off  = i*istrides[0];
                dim_type i_out_off = i*ostrides[0];

                // clear means and centers for this pixel
                for(dim_type ch=0; ch<channels; ++ch) {
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

                    for(dim_type wj=-radius; wj<=radius; ++wj) {

                        int hit_count = 0;

                        for(dim_type wi=-radius; wi<=radius; ++wi) {

                            dim_type tj = j + wj;
                            dim_type ti = i + wi;

                            // clamps offsets
                            tj = clamp(tj, 0ll, dims[1]-1);
                            ti = clamp(ti, 0ll, dims[0]-1);

                            // proceed
                            float norm = 0.0f;
                            for(dim_type ch=0; ch<channels; ++ch) {
                                tmpclrs[ch] = inData[ tj*istrides[1] + ti*istrides[0] + ch*istrides[2]];
                                norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                            }

                            if (norm<= cvar) {
                                for(dim_type ch=0; ch<channels; ++ch)
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
                    for(dim_type ch=0; ch<channels; ++ch)
                        means[ch] *= fcount;

                    float norm = 0.f;
                    for(dim_type ch=0; ch<channels; ++ch)
                        norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));
                    bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
                    shift_x = mean_x;
                    shift_y = mean_y;
                    for(dim_type ch=0; ch<channels; ++ch)
                        centers[ch] = means[ch];
                    if (stop) { break; }
                } // scope of meanshift iterations end

                for(dim_type ch=0; ch<channels; ++ch)
                    outData[j_out_off + i_out_off + ch*ostrides[2]] = centers[ch];

            }
        }

    }
    return out;
}

#define INSTANTIATE(T) \
    template Array<T> * meanshift<T, true >(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter); \
    template Array<T> * meanshift<T, false>(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
