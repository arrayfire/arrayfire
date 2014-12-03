/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

dim_type lIdx(dim_type x, dim_type y,
        dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

void load2LocalMem(__local T *  shrd,
        __global const T *      in, dim_type lx, dim_type ly,
        dim_type shrdStride, dim_type schStride, dim_type channels,
        dim_type dim0, dim_type dim1, dim_type gx, dim_type gy,
        dim_type ichStride, dim_type inStride1, dim_type inStride0)
{
    dim_type gx_  = clamp(gx, 0, dim0-1);
    dim_type gy_  = clamp(gy, 0, dim1-1);
#pragma unroll
    for(dim_type ch=0; ch<channels; ++ch)
        shrd[ lIdx(lx, ly, shrdStride, 1)+ch*schStride] = in[ lIdx(gx_, gy_, inStride1, inStride0)+ch*ichStride];
}

__kernel
void meanshift(__global T *       d_dst,
               KParam             oInfo,
               __global const T * d_src,
               KParam             iInfo,
               __local T *        localMem,
               dim_type batchIndex, dim_type channels,
               float space_, dim_type radius, float cvar,
               unsigned iter, dim_type nonBatchBlkSize)
{
    // calculate necessary offset and window parameters
    const dim_type padding     = 2*radius;
    const dim_type shrdLen     = get_local_size(0) + padding;
    const dim_type schStride   = shrdLen*(get_local_size(1) + padding);
    // the variable ichStride will only effect when we have >1
    // channels. in the other cases, the expression in question
    // will not use the variable
    const dim_type ichStride   = iInfo.strides[batchIndex-1];

    // gfor batch offsets
    unsigned batchId = get_group_id(0) / nonBatchBlkSize;
    __global const T* iptr = d_src + (batchId * iInfo.strides[batchIndex] + iInfo.offset);
    __global T*       optr = d_dst + (batchId * oInfo.strides[batchIndex]);

    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    const dim_type gx = get_local_size(0) * (get_group_id(0)-batchId*nonBatchBlkSize) + lx;
    const dim_type gy = get_local_size(1) * get_group_id(1) + ly;

    dim_type gx2 = gx + get_local_size(0);
    dim_type gy2 = gy + get_local_size(1);
    dim_type lx2 = lx + get_local_size(0);
    dim_type ly2 = ly + get_local_size(1);
    dim_type i   = lx + radius;
    dim_type j   = ly + radius;

    // pull image to local memory
    load2LocalMem(localMem, iptr, lx, ly, shrdLen, schStride, channels,
            iInfo.dims[0], iInfo.dims[1], gx-radius,
            gy-radius, ichStride, iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        load2LocalMem(localMem, iptr, lx2, ly, shrdLen, schStride, channels,
                iInfo.dims[0], iInfo.dims[1], gx2-radius,
                gy-radius, ichStride, iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        load2LocalMem(localMem, iptr, lx, ly2, shrdLen, schStride, channels,
                iInfo.dims[0], iInfo.dims[1], gx-radius,
                gy2-radius, ichStride, iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2LocalMem(localMem, iptr, lx2, ly2, shrdLen, schStride, channels,
                iInfo.dims[0], iInfo.dims[1], gx2-radius,
                gy2-radius, ichStride, iInfo.strides[1], iInfo.strides[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1])
    {
        float means[MAX_CHANNELS];
        float centers[MAX_CHANNELS];
        float tmpclrs[MAX_CHANNELS];

        // clear means and centers for this pixel
#pragma unroll
        for(dim_type ch=0; ch<channels; ++ch) {
            means[ch] = 0.0f;
            centers[ch] = localMem[lIdx(i, j, shrdLen, 1)+ch*schStride];
        }

        // scope of meanshift iterationd begin
        for(uint it=0; it<iter; ++it) {

            int count   = 0;
            int shift_x = 0;
            int shift_y = 0;

            for(dim_type wj=-radius; wj<=radius; ++wj) {
                int hit_count = 0;

                for(dim_type wi=-radius; wi<=radius; ++wi) {

                    dim_type tj = j + wj;
                    dim_type ti = i + wi;

                    // proceed
                    float norm = 0.0f;
#pragma unroll
                    for(dim_type ch=0; ch<channels; ++ch) {
                        tmpclrs[ch] = localMem[lIdx(ti, tj, shrdLen, 1)+ch*schStride];
                        norm += (centers[ch]-tmpclrs[ch]) * (centers[ch]-tmpclrs[ch]);
                    }

                    if (norm<= cvar) {
#pragma unroll
                        for(dim_type ch=0; ch<channels; ++ch)
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
            for(dim_type ch=0; ch<channels; ++ch)
                means[ch] *= fcount;

            float norm = 0.f;
#pragma unroll
            for(dim_type ch=0; ch<channels; ++ch)
                norm += ((means[ch]-centers[ch])*(means[ch]-centers[ch]));

            bool stop = ((abs(shift_y-mean_y)+abs(shift_x-mean_x)) + norm) <= 1;
            shift_x = mean_x;
            shift_y = mean_y;

#pragma unroll
            for(dim_type ch=0; ch<channels; ++ch)
                centers[ch] = means[ch];
            if (stop) { break; }
        } // scope of meanshift iterations end

#pragma unroll
        for(dim_type ch=0; ch<channels; ++ch)
            optr[lIdx(gx, gy, oInfo.strides[1], oInfo.strides[0])+ch*ichStride] = centers[ch];
    }
}
