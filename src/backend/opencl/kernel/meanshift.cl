/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

int lIdx(int x, int y,
        int stride1, int stride0)
{
    return (y*stride1 + x*stride0);
}

void load2LocalMem(__local T *  shrd,
        __global const T *      in, int lx, int ly,
        int shrdStride, int schStride, int channels,
        int dim0, int dim1, int gx, int gy,
        int ichStride, int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
#pragma unroll
    for(int ch=0; ch<channels; ++ch)
        shrd[ lIdx(lx, ly, shrdStride, 1)+ch*schStride] = in[ lIdx(gx_, gy_, inStride1, inStride0)+ch*ichStride];
}

__kernel
void meanshift(__global T *       d_dst,
               KParam             oInfo,
               __global const T * d_src,
               KParam             iInfo,
               __local T *        localMem,
               int channels, float space_, int radius,
               float cvar, unsigned iter, int nBBS0, int nBBS1)
{
    // calculate necessary offset and window parameters
    const int padding     = 2*radius + 1;
    const int wind_len    = padding - 1;
    const int shrdLen     = get_local_size(0) + padding;
    const int schStride   = shrdLen*(get_local_size(1) + padding);
    // the variable ichStride will only effect when we have >1
    // channels. in the other cases, the expression in question
    // will not use the variable
    const int ichStride   = iInfo.strides[2];

    // gfor batch offsets
    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    __global const T* iptr = d_src + (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    __global T*       optr = d_dst + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    int s0 = iInfo.strides[0];
    int s1 = iInfo.strides[1];
    int d0 = iInfo.dims[0];
    int d1 = iInfo.dims[1];
    // pull image to local memory
    for (int b=ly, gy2=gy; b<shrdLen; b+=get_local_size(1), gy2+=get_local_size(1)) {
        // move row_set get_local_size(1) along coloumns
        for (int a=lx, gx2=gx; a<shrdLen; a+=get_local_size(0), gx2+=get_local_size(0)) {
            load2LocalMem(localMem, iptr, a, b, shrdLen, schStride, channels,
                    d0, d1, gx2-radius, gy2-radius, ichStride, s1, s0);
        }
    }

    int i   = lx + radius;
    int j   = ly + radius;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1])
    {
        float means[MAX_CHANNELS];
        float centers[MAX_CHANNELS];
        float tmpclrs[MAX_CHANNELS];

        // clear means and centers for this pixel
#pragma unroll
        for(int ch=0; ch<channels; ++ch) {
            means[ch] = 0.0f;
            centers[ch] = localMem[lIdx(i, j, shrdLen, 1)+ch*schStride];
        }

        // scope of meanshift iterationd begin
        for(uint it=0; it<iter; ++it) {

            int count   = 0;
            int shift_x = 0;
            int shift_y = 0;

            for(int wj=-radius; wj<=radius; ++wj) {
                int hit_count = 0;

                for(int wi=-radius; wi<=radius; ++wi) {

                    int tj = j + wj;
                    int ti = i + wi;

                    // proceed
                    float norm = 0.0f;
#pragma unroll
                    for(int ch=0; ch<channels; ++ch) {
                        tmpclrs[ch] = localMem[lIdx(ti, tj, shrdLen, 1)+ch*schStride];
                        norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                    }

                    if (norm<= cvar) {
#pragma unroll
                        for(int ch=0; ch<channels; ++ch)
                            means[ch] += tmpclrs[ch];

                        shift_x += wi;
                        ++hit_count;
                    }
                }
                count+= hit_count;
                shift_y += wj*hit_count;
            }

            if (count==0) { break; }

            const float fcount = 1.f/count;
            const int mean_x = (int)(shift_x*fcount+0.5f);
            const int mean_y = (int)(shift_y*fcount+0.5f);
#pragma unroll
            for(int ch=0; ch<channels; ++ch)
                means[ch] *= fcount;

            float norm = 0.f;
#pragma unroll
            for(int ch=0; ch<channels; ++ch)
                norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));

            bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
            shift_x = mean_x;
            shift_y = mean_y;

#pragma unroll
            for(int ch=0; ch<channels; ++ch)
                centers[ch] = means[ch];
            if (stop) { break; }
        } // scope of meanshift iterations end

#pragma unroll
        for(int ch=0; ch<channels; ++ch)
            optr[lIdx(gx, gy, oInfo.strides[1], oInfo.strides[0])+ch*ichStride] = centers[ch];
    }
}
