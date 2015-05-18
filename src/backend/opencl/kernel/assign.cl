/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    dim_type  offs[4];
    dim_type strds[4];
    char     isSeq[4];
} AssignKernelParam_t;

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
void assignKernel(global T * optr, KParam oInfo, global const T * iptr, KParam iInfo,
                 const AssignKernelParam_t p, global const uint* ptr0,
                 global const uint* ptr1, global const uint* ptr2,
                 global const uint* ptr3, const dim_type nBBS0, const dim_type nBBS1)
{
    // retrive booleans that tell us which index to use
    const bool s0 = p.isSeq[0];
    const bool s1 = p.isSeq[1];
    const bool s2 = p.isSeq[2];
    const bool s3 = p.isSeq[3];

    const dim_type gz = get_group_id(0)/nBBS0;
    const dim_type gw = get_group_id(1)/nBBS1;
    const dim_type gx = get_local_size(0) * (get_group_id(0) - gz*nBBS0) + get_local_id(0);
    const dim_type gy = get_local_size(1) * (get_group_id(1) - gw*nBBS1) + get_local_id(1);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1] && gz<iInfo.dims[2] && gw<iInfo.dims[3]) {
        // calculate pointer offsets for input
        dim_type i = p.strds[0] * trimIndex(s0 ? gx+p.offs[0] : ptr0[gx], oInfo.dims[0]);
        dim_type j = p.strds[1] * trimIndex(s1 ? gy+p.offs[1] : ptr1[gy], oInfo.dims[1]);
        dim_type k = p.strds[2] * trimIndex(s2 ? gz+p.offs[2] : ptr2[gz], oInfo.dims[2]);
        dim_type l = p.strds[3] * trimIndex(s3 ? gw+p.offs[3] : ptr3[gw], oInfo.dims[3]);
        // offset input and output pointers
        global const T *src = iptr + (gx*iInfo.strides[0]+
                                      gy*iInfo.strides[1]+
                                      gz*iInfo.strides[2]+
                                      gw*iInfo.strides[3]);
        global T *dst = optr + (i+j+k+l);
        // set the output
        dst[0] = src[0];
    }
}
