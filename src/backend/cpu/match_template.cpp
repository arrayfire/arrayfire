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
#include <match_template.hpp>
#include <err_cpu.hpp>

using af::dim4;

namespace cpu
{

template<typename inType, typename outType, af_match_type mType>
Array<outType> match_template(const Array<inType> &sImg, const Array<inType> &tImg)
{
    const dim4 sDims = sImg.dims();
    const dim4 tDims = tImg.dims();
    const dim4 sStrides = sImg.strides();
    const dim4 tStrides = tImg.strides();

    const dim_type tDim0  = tDims[0];
    const dim_type tDim1  = tDims[1];
    const dim_type sDim0  = sDims[0];
    const dim_type sDim1  = sDims[1];

    Array<outType> out = createEmptyArray<outType>(sDims);
    const dim4 oStrides = out.strides();

    const dim_type batchNum = sDims[2];

    outType tImgMean = outType(0);
    dim_type winNumElements = tImg.elements();
    bool needMean = mType==AF_ZSAD || mType==AF_LSAD ||
                    mType==AF_ZSSD || mType==AF_LSSD ||
                    mType==AF_ZNCC;
    const inType * tpl = tImg.get();

    if (needMean) {
        for(dim_type tj=0; tj<tDim1; tj++) {
            dim_type tjStride = tj*tStrides[1];

            for(dim_type ti=0; ti<tDim0; ti++) {
                tImgMean += (outType)tpl[tjStride+ti*tStrides[0]];
            }
        }
        tImgMean /= winNumElements;
    }

    for(dim_type b=0; b<batchNum; ++b) {
        outType * dst      = out.get() + b*oStrides[2];
        const inType * src = sImg.get() + b*sStrides[2];

        // slide through image window after window
        for(dim_type sj=0; sj<sDim1; sj++) {

            dim_type ojStride = sj*oStrides[1];

            for(dim_type si=0; si<sDim0; si++) {
                outType disparity = outType(0);

                // mean for window
                // this variable will be used based on mType value
                outType wImgMean = outType(0);
                if (needMean) {
                    for(dim_type tj=0,j=sj; tj<tDim1; tj++, j++) {
                        dim_type jStride = j*sStrides[1];

                        for(dim_type ti=0, i=si; ti<tDim0; ti++, i++) {
                            inType sVal = ((j<sDim1 && i<sDim0) ?
                                    src[jStride + i*sStrides[0]] : inType(0));
                            wImgMean += (outType)sVal;
                        }
                    }
                    wImgMean /= winNumElements;
                }

                // run the window match metric
                for(dim_type tj=0,j=sj; tj<tDim1; tj++, j++) {
                    dim_type jStride = j*sStrides[1];
                    dim_type tjStride = tj*tStrides[1];

                    for(dim_type ti=0, i=si; ti<tDim0; ti++, i++) {
                        inType sVal = ((j<sDim1 && i<sDim0) ?
                                            src[jStride + i*sStrides[0]] : inType(0));
                        inType tVal = tpl[tjStride+ti*tStrides[0]];
                        outType temp;
                        switch(mType) {
                            case AF_SAD:
                                disparity += fabs((outType)sVal-(outType)tVal);
                                break;
                            case AF_ZSAD:
                                disparity += fabs((outType)sVal - wImgMean -
                                                  (outType)tVal + tImgMean);
                                break;
                            case AF_LSAD:
                                disparity += fabs((outType)sVal-(wImgMean/tImgMean)*tVal);
                                break;
                            case AF_SSD:
                                disparity += ((outType)sVal-(outType)tVal)*((outType)sVal-(outType)tVal);
                                break;
                            case AF_ZSSD:
                                temp = ((outType)sVal - wImgMean - (outType)tVal + tImgMean);
                                disparity += temp*temp;
                                break;
                            case AF_LSSD:
                                temp = ((outType)sVal-(wImgMean/tImgMean)*tVal);
                                disparity += temp*temp;
                                break;
                            case AF_NCC:
                                //TODO: furture implementation
                                break;
                            case AF_ZNCC:
                                //TODO: furture implementation
                                break;
                            case AF_SHD:
                                //TODO: furture implementation
                                break;
                        }
                    }
                }
                // output is just created, hence not doing the
                // extra multiplication for 0th dim stride
                dst[ojStride + si] = disparity;
            }
        }
    }

    return out;
}

#define INSTANTIATE(in_t, out_t)\
    template Array<out_t> match_template<in_t, out_t, AF_SAD >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_LSAD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZSAD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_SSD >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_LSSD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZSSD>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_NCC >(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_ZNCC>(const Array<in_t> &sImg, const Array<in_t> &tImg); \
    template Array<out_t> match_template<in_t, out_t, AF_SHD >(const Array<in_t> &sImg, const Array<in_t> &tImg);

INSTANTIATE(double, double)
INSTANTIATE(float ,  float)
INSTANTIATE(char  ,  float)
INSTANTIATE(int   ,  float)
INSTANTIATE(uint  ,  float)
INSTANTIATE(uchar ,  float)

}
