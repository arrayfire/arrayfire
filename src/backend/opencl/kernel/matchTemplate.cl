/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel
void matchTemplate(global outType * out,
                    KParam          oInfo,
                    global const inType * srch,
                    KParam          sInfo,
                    global const inType * tmplt,
                    KParam          tInfo,
                    dim_type        nBBS0,
                    dim_type        nBBS1)
{
    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;

    dim_type gx = get_local_id(0) + (get_group_id(0) - b2*nBBS0) * get_local_size(0);
    dim_type gy = get_local_id(1) + (get_group_id(1) - b3*nBBS1)* get_local_size(1);

    if (gx < sInfo.dims[0] && gy < sInfo.dims[1]) {

        const dim_type tDim0 = tInfo.dims[0];
        const dim_type tDim1 = tInfo.dims[1];
        const dim_type sDim0 = sInfo.dims[0];
        const dim_type sDim1 = sInfo.dims[1];
        dim_type winNumElems = tDim0*tDim1;

        global const inType* tptr = tmplt;

        outType tImgMean = (outType)0;
        if (NEEDMEAN) {
            for(dim_type tj=0; tj<tDim1; tj++) {
                dim_type tjStride = tj*tInfo.strides[1];

                for(dim_type ti=0; ti<tDim0; ti++) {
                    tImgMean += (outType)tptr[ tjStride + ti*tInfo.strides[0] ];
                }
            }
            tImgMean /= winNumElems;
        }

        global const inType* sptr  = srch + (b2 * sInfo.strides[2] + b3 * sInfo.strides[3] + sInfo.offset);
        global outType* optr       = out  + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

        // mean for window
        // this variable will be used based on MATCH_T value
        outType wImgMean = (outType)0;
        if (NEEDMEAN) {
            for(dim_type tj=0,j=gy; tj<tDim1; tj++, j++) {
                dim_type jStride = j*sInfo.strides[1];

                for(dim_type ti=0, i=gx; ti<tDim0; ti++, i++) {
                    inType sVal = ((j<sDim1 && i<sDim0) ? sptr[jStride + i*sInfo.strides[0]] : (inType)0);
                    wImgMean += (outType)sVal;
                }
            }
            wImgMean /= winNumElems;
        }

        // run the window match metric
        outType disparity = (outType)0;

        for(dim_type tj=0,j=gy; tj<tDim1; tj++, j++) {

            dim_type jStride  = j*sInfo.strides[1];
            dim_type tjStride = tj*tInfo.strides[1];

            for(dim_type ti=0, i=gx; ti<tDim0; ti++, i++) {

                inType sVal = ((j<sDim1 && i<sDim0) ? sptr[jStride + i*sInfo.strides[0]] : (inType)0);
                inType tVal = tptr[ tjStride + ti*tInfo.strides[0] ];

                outType temp;
                switch(MATCH_T) {
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

        optr[gy*oInfo.strides[1]+gx] = disparity;
    }
}
