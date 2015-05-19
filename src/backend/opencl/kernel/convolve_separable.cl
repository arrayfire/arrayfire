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
              KParam sInfo, constant accType const *impulse,
              int nBBS0, int nBBS1)
{
    local T localMem[LOCAL_MEM_SIZE];

    const int radius  = FLEN-1;
    const int padding = 2*radius;
    const int s0      = sInfo.strides[0];
    const int s1      = sInfo.strides[1];
    const int d0      = sInfo.dims[0];
    const int d1      = sInfo.dims[1];
    const int shrdLen = get_local_size(0) + (CONV_DIM==0 ? padding : 0);

    unsigned b2  = get_group_id(0)/nBBS0;
    unsigned b3  = get_group_id(1)/nBBS1;
    global T *dst = out + (b2*oInfo.strides[2] + b3*oInfo.strides[3]);
    global const T *src = signal + (b2*sInfo.strides[2] + b3*sInfo.strides[3]) + sInfo.offset;

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int ox = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    int oy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;
    int gx = ox;
    int gy = oy;

    // below if-else statement is based on MACRO value passed while kernel compilation
    if (CONV_DIM==0) {
        gx += (EXPAND ? 0 : FLEN>>1);
        int endX = ((FLEN-1)<<1) + get_local_size(0);
#pragma unroll
        for(int lx = get_local_id(0), glb_x = gx; lx<endX; lx += get_local_size(0), glb_x += get_local_size(0)) {
            int i = glb_x - radius;
            int j = gy;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            localMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : (T)(0));
        }

    } else if (CONV_DIM==1) {
        gy += (EXPAND ? 0 : FLEN>>1);
        int endY = ((FLEN-1)<<1) + get_local_size(1);
#pragma unroll
        for(int ly = get_local_id(1), glb_y = gy; ly<endY; ly += get_local_size(1), glb_y += get_local_size(1)) {
            int i = gx;
            int j = glb_y - radius;
            bool is_i  = i>=0 && i<d0;
            bool is_j  = j>=0 && j<d1;
            localMem[ly*shrdLen+lx] = (is_i && is_j ? src[i*s0 + j*s1] : (T)(0));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ox<oInfo.dims[0] && oy<oInfo.dims[1]) {
        // below conditional statement is based on MACRO value passed while kernel compilation
        int i  = (CONV_DIM==0 ? lx : ly) + radius;
        accType accum = (accType)(0);
#pragma unroll
        for(int f=0; f<FLEN; ++f) {
            accType f_val = impulse[f];
            // below conditional statement is based on MACRO value passed while kernel compilation
            int s_idx = (CONV_DIM==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = localMem[s_idx];
            accum   = accum + ((accType)s_val*(accType)f_val);
        }
        dst[oy*oInfo.strides[1]+ox] = (T)accum;
    }
}
