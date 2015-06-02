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
#include <morph.hpp>
#include <algorithm>

using af::dim4;

namespace cpu
{

static inline unsigned getIdx(const dim4 &strides,
        int i, int j = 0, int k = 0, int l = 0)
{
    return (l * strides[3] +
            k * strides[2] +
            j * strides[1] +
            i * strides[0]);
}

template<typename T, bool isDilation>
Array<T> morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 dims       = in.dims();
    const dim4 window     = mask.dims();
    const dim_t R0     = window[0]/2;
    const dim_t R1     = window[1]/2;
    const dim4 istrides   = in.strides();
    const dim4 fstrides   = mask.strides();

    Array<T> out         = createEmptyArray<T>(dims);
    const dim4 ostrides   = out.strides();

    T* outData            = out.get();
    const T*   inData     = in.get();
    const T*   filter     = mask.get();

    for(dim_t b3=0; b3<dims[3]; ++b3) {
        for(dim_t b2=0; b2<dims[2]; ++b2) {
            // either channels or batch is handled by outer most loop
            for(dim_t j=0; j<dims[1]; ++j) {
                // j steps along 2nd dimension
                for(dim_t i=0; i<dims[0]; ++i) {
                    // i steps along 1st dimension
                    T filterResult = inData[ getIdx(istrides, i, j) ];

                    // wj,wi steps along 2nd & 1st dimensions of filter window respectively
                    for(dim_t wj=0; wj<window[1]; wj++) {
                        for(dim_t wi=0; wi<window[0]; wi++) {

                            dim_t offj = j+wj-R1;
                            dim_t offi = i+wi-R0;

                            T maskValue = filter[ getIdx(fstrides, wi, wj) ];

                            if ((maskValue > (T)0) && offi>=0 && offj>=0 && offi<dims[0] && offj<dims[1]) {

                                T inValue   = inData[ getIdx(istrides, offi, offj) ];

                                if (isDilation)
                                    filterResult = std::max(filterResult, inValue);
                                else
                                    filterResult = std::min(filterResult, inValue);
                            }

                        } // window 1st dimension loop ends here
                    } // filter window loop ends here

                    outData[ getIdx(ostrides, i, j) ] = filterResult;
                } //1st dimension loop ends here
            } // 2nd dimension loop ends here

            // next iteration will be next batch if any
            outData += ostrides[2];
            inData  += istrides[2];
        }
    }

    return out;
}

template<typename T, bool isDilation>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 dims       = in.dims();
    const dim4 window     = mask.dims();
    const dim_t R0     = window[0]/2;
    const dim_t R1     = window[1]/2;
    const dim_t R2     = window[2]/2;
    const dim4 istrides   = in.strides();
    const dim4 fstrides   = mask.strides();
    const dim_t bCount = dims[3];

    Array<T> out         = createEmptyArray<T>(dims);
    const dim4 ostrides   = out.strides();

    T* outData            = out.get();
    const T*   inData     = in.get();
    const T*   filter     = mask.get();

    for(dim_t batchId=0; batchId<bCount; ++batchId) {
        // either channels or batch is handled by outer most loop
        for(dim_t k=0; k<dims[2]; ++k) {
            // k steps along 3rd dimension
            for(dim_t j=0; j<dims[1]; ++j) {
                // j steps along 2nd dimension
                for(dim_t i=0; i<dims[0]; ++i) {
                    // i steps along 1st dimension
                    T filterResult = inData[ getIdx(istrides, i, j, k) ];

                    // wk, wj,wi steps along 2nd & 1st dimensions of filter window respectively
                    for(dim_t wk=0; wk<window[2]; wk++) {
                        for(dim_t wj=0; wj<window[1]; wj++) {
                            for(dim_t wi=0; wi<window[0]; wi++) {

                                dim_t offk = k+wk-R2;
                                dim_t offj = j+wj-R1;
                                dim_t offi = i+wi-R0;

                                T maskValue = filter[ getIdx(fstrides, wi, wj, wk) ];

                                if ((maskValue > (T)0) && offi>=0 && offj>=0 && offk>=0 &&
                                        offi<dims[0] && offj<dims[1] && offk<dims[2]) {

                                    T inValue   = inData[ getIdx(istrides, offi, offj, offk) ];

                                    if (isDilation)
                                        filterResult = std::max(filterResult, inValue);
                                    else
                                        filterResult = std::min(filterResult, inValue);
                                }

                            } // window 1st dimension loop ends here
                        }  // window 1st dimension loop ends here
                    }// filter window loop ends here

                    outData[ getIdx(ostrides, i, j, k) ] = filterResult;
                } //1st dimension loop ends here
            } // 2nd dimension loop ends here
        } // 3rd dimension loop ends here
        // next iteration will be next batch if any
        outData += ostrides[3];
        inData  += istrides[3];
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> morph  <T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph  <T, false>(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph3d<T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> morph3d<T, false>(const Array<T> &in, const Array<T> &mask);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
