/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

template<typename inType, typename outType, af_match_type mType, bool needMean>
__global__
void matchTemplate(Param<outType> out, CParam<inType> srch, CParam<inType> tmplt, dim_type nBBS)
{
    unsigned batchId = blockIdx.x / nBBS;

    dim_type gx = threadIdx.x + (blockIdx.x - batchId*nBBS) * blockDim.x;
    dim_type gy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gx < srch.dims[0] && gy < srch.dims[1]) {

        const dim_type tDim0 = tmplt.dims[0];
        const dim_type tDim1 = tmplt.dims[1];
        const dim_type sDim0 = srch.dims[0];
        const dim_type sDim1 = srch.dims[1];
        const inType* tptr   = (const inType*) tmplt.ptr;
        dim_type winNumElems = tDim0*tDim1;

        outType tImgMean = outType(0);
        if (needMean) {
            for(dim_type tj=0; tj<tDim1; tj++) {
                dim_type tjStride = tj*tmplt.strides[1];

                for(dim_type ti=0; ti<tDim0; ti++) {
                    tImgMean += (outType)tptr[ tjStride + ti*tmplt.strides[0] ];
                }
            }
            tImgMean /= winNumElems;
        }

        const inType* sptr  = (const inType*) srch.ptr + (batchId * srch.strides[2]);
        outType* optr       = (outType*) out.ptr + (batchId * out.strides[2]);

        // mean for window
        // this variable will be used based on mType value
        outType wImgMean = outType(0);
        if (needMean) {
            for(dim_type tj=0,j=gy; tj<tDim1; tj++, j++) {
                dim_type jStride = j*srch.strides[1];

                for(dim_type ti=0, i=gx; ti<tDim0; ti++, i++) {
                    inType sVal = ((j<sDim1 && i<sDim0) ? sptr[jStride + i*srch.strides[0]] : inType(0));
                    wImgMean += (outType)sVal;
                }
            }
            wImgMean /= winNumElems;
        }

        // run the window match metric
        outType disparity = outType(0);

        for(dim_type tj=0,j=gy; tj<tDim1; tj++, j++) {

            dim_type jStride  = j*srch.strides[1];
            dim_type tjStride = tj*tmplt.strides[1];

            for(dim_type ti=0, i=gx; ti<tDim0; ti++, i++) {

                inType sVal = ((j<sDim1 && i<sDim0) ? sptr[jStride + i*srch.strides[0]] : inType(0));
                inType tVal = tptr[ tjStride + ti*tmplt.strides[0] ];

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

        optr[gy*out.strides[1]+gx] = disparity;
    }
}

template<typename inType, typename outType, af_match_type mType, bool needMean>
void matchTemplate(Param<outType> out, CParam<inType> srch, CParam<inType> tmplt)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(srch.dims[0], threads.x);
    dim_type blk_y = divup(srch.dims[1], threads.y);

    dim3 blocks(blk_x*srch.dims[2], blk_y);

    matchTemplate<inType, outType, mType, needMean> <<< blocks, threads >>> (out, srch, tmplt, blk_x);

    POST_LAUNCH_CHECK();
}

}

}
