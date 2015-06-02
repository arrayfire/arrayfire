/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    int  offs[4];
    int strds[4];
    char     isSeq[4];
} IndexKernelParam_t;

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
void indexKernel(global T * optr, KParam oInfo, global const T * iptr, KParam iInfo,
                 const IndexKernelParam_t p, global const uint* ptr0,
                 global const uint* ptr1, global const uint* ptr2,
                 global const uint* ptr3, const int nBBS0, const int nBBS1)
{
    // retrive booleans that tell us which index to use
    const bool s0 = p.isSeq[0];
    const bool s1 = p.isSeq[1];
    const bool s2 = p.isSeq[2];
    const bool s3 = p.isSeq[3];

    const int gz = get_group_id(0)/nBBS0;
    const int gw = get_group_id(1)/nBBS1;
    const int gx = get_local_size(0) * (get_group_id(0) - gz*nBBS0) + get_local_id(0);
    const int gy = get_local_size(1) * (get_group_id(1) - gw*nBBS1) + get_local_id(1);

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1] && gz<oInfo.dims[2] && gw<oInfo.dims[3]) {
        // calculate pointer offsets for input
        int i = p.strds[0] * trimIndex(s0 ? gx+p.offs[0] : ptr0[gx], iInfo.dims[0]);
        int j = p.strds[1] * trimIndex(s1 ? gy+p.offs[1] : ptr1[gy], iInfo.dims[1]);
        int k = p.strds[2] * trimIndex(s2 ? gz+p.offs[2] : ptr2[gz], iInfo.dims[2]);
        int l = p.strds[3] * trimIndex(s3 ? gw+p.offs[3] : ptr3[gw], iInfo.dims[3]);
        // offset input and output pointers
        global const T *src = iptr + (i+j+k+l);
        global T *dst = optr + (gx*oInfo.strides[0]+
                                gy*oInfo.strides[1]+
                                gz*oInfo.strides[2]+
                                gw*oInfo.strides[3]);
        // set the output
        dst[0] = src[0];
    }
}
