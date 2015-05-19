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

    const dim_t tDim0  = tDims[0];
    const dim_t tDim1  = tDims[1];
    const dim_t sDim0  = sDims[0];
    const dim_t sDim1  = sDims[1];

    Array<outType> out = createEmptyArray<outType>(sDims);
    const dim4 oStrides = out.strides();

    outType tImgMean = outType(0);
    dim_t winNumElements = tImg.elements();
    bool needMean = mType==AF_ZSAD || mType==AF_LSAD ||
                    mType==AF_ZSSD || mType==AF_LSSD ||
                    mType==AF_ZNCC;
    const inType * tpl = tImg.get();

    if (needMean) {
        for(dim_t tj=0; tj<tDim1; tj++) {
            dim_t tjStride = tj*tStrides[1];

            for(dim_t ti=0; ti<tDim0; ti++) {
                tImgMean += (outType)tpl[tjStride+ti*tStrides[0]];
            }
        }
        tImgMean /= winNumElements;
    }

    outType * dst      = out.get();
    const inType * src = sImg.get();

    for(dim_t b3=0; b3<sDims[3]; ++b3) {
    for(dim_t b2=0; b2<sDims[2]; ++b2) {

        // slide through image window after window
        for(dim_t sj=0; sj<sDim1; sj++) {

            dim_t ojStride = sj*oStrides[1];

            for(dim_t si=0; si<sDim0; si++) {
                outType disparity = outType(0);

                // mean for window
                // this variable will be used based on mType value
                outType wImgMean = outType(0);
                if (needMean) {
                    for(dim_t tj=0,j=sj; tj<tDim1; tj++, j++) {
                        dim_t jStride = j*sStrides[1];

                        for(dim_t ti=0, i=si; ti<tDim0; ti++, i++) {
                            inType sVal = ((j<sDim1 && i<sDim0) ?
                                    src[jStride + i*sStrides[0]] : inType(0));
                            wImgMean += (outType)sVal;
                        }
                    }
                    wImgMean /= winNumElements;
                }

                // run the window match metric
                for(dim_t tj=0,j=sj; tj<tDim1; tj++, j++) {
                    dim_t jStride = j*sStrides[1];
                    dim_t tjStride = tj*tStrides[1];

                    for(dim_t ti=0, i=si; ti<tDim0; ti++, i++) {
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
        src += sStrides[2];
        dst += oStrides[2];
    }
        src += sStrides[3];
        dst += oStrides[3];
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
