/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void meanshift(__global T *       d_dst,
               KParam             oInfo,
               __global const T * d_src,
               KParam             iInfo,
               int radius, float cvar, unsigned iter,
               int nBBS0, int nBBS1)
{
    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    __global const T* iptr = d_src + (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    __global T*       optr = d_dst + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + get_local_id(0);
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + get_local_id(1);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1])
    {
        int i  = gx;
        int j  = gy;

        T centers[MAX_CHANNELS];
        T tmpclrs[MAX_CHANNELS];

        AccType means[MAX_CHANNELS];

#pragma unroll
        for(int ch=0; ch<MAX_CHANNELS; ++ch)
            centers[ch] = iptr[ i*iInfo.strides[0]+j*iInfo.strides[1]+ch*iInfo.strides[2] ];

        const int dim0LenLmt = iInfo.dims[0]-1;
        const int dim1LenLmt = iInfo.dims[1]-1;

        // scope of meanshift iterationd begin
        for(uint it=0; it<iter; ++it) {

            int ocj = j;
            int oci = i;
            unsigned count   = 0;

            int shift_x = 0;
            int shift_y = 0;

            for (int ch=0; ch<MAX_CHANNELS; ++ch)
                means[ch] = 0;

            for(int wj=-radius; wj<=radius; ++wj) {
                int hit_count = 0;
                int tj = j + wj;

                if (tj<0 || tj>dim1LenLmt) continue;

                for(int wi=-radius; wi<=radius; ++wi) {

                    int ti = i + wi;

                    if (ti<0 || ti>dim0LenLmt) continue;

                    AccType norm = 0;
#pragma unroll
                    for(int ch=0; ch<MAX_CHANNELS; ++ch) {
                        unsigned idx = ti*iInfo.strides[0] + tj*iInfo.strides[1] + ch*iInfo.strides[2];
                        tmpclrs[ch] = iptr[idx];
                        AccType diff = (AccType)centers[ch] - (AccType)tmpclrs[ch];
                        norm += (diff * diff);
                    }

                    if (norm<= cvar) {
#pragma unroll
                        for(int ch=0; ch<MAX_CHANNELS; ++ch)
                            means[ch] += (AccType)tmpclrs[ch];

                        shift_x += ti;
                        ++hit_count;
                    }
                }
                count+= hit_count;
                shift_y += tj*hit_count;
            }

            if (count==0) break;

            const AccType fcount = 1/(AccType)count;

            i = convert_int_rtz(shift_x*fcount);
            j = convert_int_rtz(shift_y*fcount);

#pragma unroll
            for(int ch=0; ch<MAX_CHANNELS; ++ch)
                means[ch] = convert_int_rtz(means[ch]*fcount);

            AccType norm = 0;
#pragma unroll
            for(int ch=0; ch<MAX_CHANNELS; ++ch) {
                AccType diff = (AccType)centers[ch] - means[ch];
                norm += (diff*diff);
            }

            bool stop = (j==ocj && i==oci) || ((abs(ocj-j)+abs(oci-i)) + norm) <= 1;

#pragma unroll
            for(int ch=0; ch<MAX_CHANNELS; ++ch)
                centers[ch] = (T)(means[ch]);

            if (stop) break;
        } // scope of meanshift iterations end

#pragma unroll
        for(int ch=0; ch<MAX_CHANNELS; ++ch)
            optr[gx*oInfo.strides[0] + gy*oInfo.strides[1] + ch*oInfo.strides[2]] = centers[ch];
    }
}
