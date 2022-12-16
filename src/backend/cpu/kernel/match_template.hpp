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

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename OutT, typename InT, af::matchType MatchType>
void matchTemplate(Param<OutT> out, CParam<InT> sImg, CParam<InT> tImg) {
    constexpr bool needMean = MatchType == AF_ZSAD || MatchType == AF_LSAD ||
                              MatchType == AF_ZSSD || MatchType == AF_LSSD ||
                              MatchType == AF_ZNCC;

    const af::dim4 sDims    = sImg.dims();
    const af::dim4 tDims    = tImg.dims();
    const af::dim4 sStrides = sImg.strides();
    const af::dim4 tStrides = tImg.strides();

    const dim_t tDim0 = tDims[0];
    const dim_t tDim1 = tDims[1];
    const dim_t sDim0 = sDims[0];
    const dim_t sDim1 = sDims[1];

    const af::dim4 oStrides = out.strides();

    OutT tImgMean        = OutT(0);
    dim_t winNumElements = tImg.dims().elements();
    const InT* tpl       = tImg.get();

    if (needMean) {
        for (dim_t tj = 0; tj < tDim1; tj++) {
            dim_t tjStride = tj * tStrides[1];

            for (dim_t ti = 0; ti < tDim0; ti++) {
                tImgMean += (OutT)tpl[tjStride + ti * tStrides[0]];
            }
        }
        tImgMean /= winNumElements;
    }

    OutT* dst      = out.get();
    const InT* src = sImg.get();

    for (dim_t b3 = 0; b3 < sDims[3]; ++b3) {
        for (dim_t b2 = 0; b2 < sDims[2]; ++b2) {
            // slide through image window after window
            for (dim_t sj = 0; sj < sDim1; sj++) {
                dim_t ojStride = sj * oStrides[1];

                for (dim_t si = 0; si < sDim0; si++) {
                    OutT disparity = OutT(0);

                    // mean for window
                    // this variable will be used based on MatchType value
                    OutT wImgMean = OutT(0);
                    if (needMean) {
                        for (dim_t tj = 0, j = sj; tj < tDim1; tj++, j++) {
                            dim_t jStride = j * sStrides[1];

                            for (dim_t ti = 0, i = si; ti < tDim0; ti++, i++) {
                                InT sVal = ((j < sDim1 && i < sDim0)
                                                ? src[jStride + i * sStrides[0]]
                                                : InT(0));
                                wImgMean += (OutT)sVal;
                            }
                        }
                        wImgMean /= winNumElements;
                    }

                    // run the window match metric
                    for (dim_t tj = 0, j = sj; tj < tDim1; tj++, j++) {
                        dim_t jStride  = j * sStrides[1];
                        dim_t tjStride = tj * tStrides[1];

                        for (dim_t ti = 0, i = si; ti < tDim0; ti++, i++) {
                            InT sVal = ((j < sDim1 && i < sDim0)
                                            ? src[jStride + i * sStrides[0]]
                                            : InT(0));
                            InT tVal = tpl[tjStride + ti * tStrides[0]];
                            OutT temp;
                            switch (MatchType) {
                                case AF_SAD:
                                    disparity += fabs((OutT)sVal - (OutT)tVal);
                                    break;
                                case AF_ZSAD:
                                    disparity += fabs((OutT)sVal - wImgMean -
                                                      (OutT)tVal + tImgMean);
                                    break;
                                case AF_LSAD:
                                    disparity +=
                                        fabs((OutT)sVal -
                                             (wImgMean / tImgMean) * tVal);
                                    break;
                                case AF_SSD:
                                    disparity += ((OutT)sVal - (OutT)tVal) *
                                                 ((OutT)sVal - (OutT)tVal);
                                    break;
                                case AF_ZSSD:
                                    temp = ((OutT)sVal - wImgMean - (OutT)tVal +
                                            tImgMean);
                                    disparity += temp * temp;
                                    break;
                                case AF_LSSD:
                                    temp = ((OutT)sVal -
                                            (wImgMean / tImgMean) * tVal);
                                    disparity += temp * temp;
                                    break;
                                case AF_NCC:
                                    // TODO: furture implementation
                                    break;
                                case AF_ZNCC:
                                    // TODO: furture implementation
                                    break;
                                case AF_SHD:
                                    // TODO: furture implementation
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
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
