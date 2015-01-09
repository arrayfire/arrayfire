/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

dim_type trimIndex(dim_type idx, const dim_type len)
{
    dim_type ret_val = idx;
    dim_type offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

kernel
void arrayIndexND(global in_t * out,
                  KParam oInfo,
                  global const in_t * in,
                  KParam iInfo,
                  global const idx_t * indices,
                  KParam idxInfo,
                  dim_type nBBS0,
                  dim_type nBBS1)
{
    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);

    dim_type gz = get_group_id(0)/nBBS0;
    dim_type gw = get_group_id(1)/nBBS1;

    dim_type gx = get_num_groups(0) * (get_group_id(0) - gz*nBBS0) + lx;
    dim_type gy = get_num_groups(1) * (get_group_id(1) - gw*nBBS1) + ly;

    global const idx_t *idxPtr = indices;

    dim_type i = iInfo.strides[0]*(DIM==0 ? trimIndex((dim_type)idxPtr[gx], iInfo.dims[0]): gx);
    dim_type j = iInfo.strides[1]*(DIM==1 ? trimIndex((dim_type)idxPtr[gy], iInfo.dims[1]): gy);
    dim_type k = iInfo.strides[2]*(DIM==2 ? trimIndex((dim_type)idxPtr[gz], iInfo.dims[2]): gz);
    dim_type l = iInfo.strides[3]*(DIM==3 ? trimIndex((dim_type)idxPtr[gw], iInfo.dims[3]): gw);

    global const in_t *inPtr = in + (i+j+k+l);
    global in_t *outPtr = out + (gx*oInfo.strides[0]+gy*oInfo.strides[1]+
                                 gz*oInfo.strides[2]+gw*oInfo.strides[3]);

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1] && gz<oInfo.dims[2] && gw<oInfo.dims[3]) {
        outPtr[0] = inPtr[0];
    }
}
