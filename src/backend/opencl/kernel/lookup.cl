/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

int trimIndex(int idx, const int len)
{
    int ret_val = idx;
    int offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

kernel
void lookupND(global in_t * out,
                  KParam oInfo,
                  global const in_t * in,
                  KParam iInfo,
                  global const idx_t * indices,
                  KParam idxInfo,
                  int nBBS0,
                  int nBBS1)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gz = get_group_id(0)/nBBS0;
    int gw = get_group_id(1)/nBBS1;

    int gx = get_local_size(0) * (get_group_id(0) - gz*nBBS0) + lx;
    int gy = get_local_size(1) * (get_group_id(1) - gw*nBBS1) + ly;

    global const idx_t *idxPtr = indices;

    int i = iInfo.strides[0]*(DIM==0 ? trimIndex((int)idxPtr[gx], iInfo.dims[0]): gx);
    int j = iInfo.strides[1]*(DIM==1 ? trimIndex((int)idxPtr[gy], iInfo.dims[1]): gy);
    int k = iInfo.strides[2]*(DIM==2 ? trimIndex((int)idxPtr[gz], iInfo.dims[2]): gz);
    int l = iInfo.strides[3]*(DIM==3 ? trimIndex((int)idxPtr[gw], iInfo.dims[3]): gw);

    global const in_t *inPtr = in + (i+j+k+l);
    global in_t *outPtr = out + (gx*oInfo.strides[0]+gy*oInfo.strides[1]+
                                 gz*oInfo.strides[2]+gw*oInfo.strides[3]);

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1] && gz<oInfo.dims[2] && gw<oInfo.dims[3]) {
        outPtr[0] = inPtr[0];
    }
}
