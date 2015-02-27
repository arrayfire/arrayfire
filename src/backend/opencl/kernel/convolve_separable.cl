/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel
void convolve(global T *out, KParam oInfo, global T const *signal,
              KParam sInfo, constant T const *impulse, dim_type nBBS)
{
    local T localMem[LOCAL_MEM_SIZE];

    const dim_type radius  = FLEN-1;
    const dim_type padding = 2*radius;
    const dim_type s0      = sInfo.strides[0];
    const dim_type s1      = sInfo.strides[1];
    const dim_type d0      = sInfo.dims[0];
    const dim_type d1      = sInfo.dims[1];
    const dim_type shrdLen = get_local_size(0) + (CONV_DIM==0 ? padding : 0);

    unsigned batchId  = get_group_id(0)/nBBS;
    global T *dst = out + (batchId*oInfo.strides[2]);
    global const T *src = signal + (batchId*sInfo.strides[2]) + sInfo.offset;

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);
    dim_type ox = get_local_size(0) * (get_group_id(0)-batchId*nBBS) + lx;
    dim_type oy = get_local_size(1) * get_group_id(1) + ly;
    dim_type gx = ox;
    dim_type gy = oy;

    // below if-else statement is based on MACRO value passed while kernel compilation
    if (CONV_DIM==0) {
        gx += (EXPAND ? 0 : FLEN>>1);
        dim_type endX = ((FLEN-1)<<1) + get_local_size(0);
#pragma unroll
        for(dim_type lx = get_local_id(0), glb_x = gx; lx<endX; lx += get_local_size(0), glb_x += get_local_size(0)) {
            dim_type i = glb_x - radius;
            dim_type j = gy;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            localMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : (T)(0));
        }

    } else if (CONV_DIM==1) {
        gy += (EXPAND ? 0 : FLEN>>1);
        dim_type endY = ((FLEN-1)<<1) + get_local_size(1);
#pragma unroll
        for(dim_type ly = get_local_id(1), glb_y = gy; ly<endY; ly += get_local_size(1), glb_y += get_local_size(1)) {
            dim_type i = gx;
            dim_type j = glb_y - radius;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            localMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : (T)(0));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ox<oInfo.dims[0] && oy<oInfo.dims[1]) {
        // below conditional statement is based on MACRO value passed while kernel compilation
        dim_type i  = (CONV_DIM==0 ? lx : ly) + radius;
        accType accum = (accType)(0);
#pragma unroll
        for(dim_type f=0; f<FLEN; ++f) {
            T f_val = impulse[f];
            // below conditional statement is based on MACRO value passed while kernel compilation
            dim_type s_idx = (CONV_DIM==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = localMem[s_idx];
            accum   = accum + ((accType)s_val*(accType)f_val);
        }
        dst[oy*oInfo.strides[1]+ox] = (T)accum;
    }
}
