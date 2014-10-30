/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

dim_type lIdx(dim_type x, dim_type y,
        dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

void load2LocalMem(__local outType *  shrd,
        __global const inType *      in,
        dim_type lx, dim_type ly, dim_type shrdStride,
        dim_type dim0, dim_type dim1,
        dim_type gx, dim_type gy,
        dim_type inStride1, dim_type inStride0)
{
    dim_type gx_  = clamp(gx, (long)0, dim0-1);
    dim_type gy_  = clamp(gy, (long)0, dim1-1);
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = (outType)in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

float gaussian1d(float x, float variance)
{
    const float exponent = (x * x) / (-2.f * variance);
    return exp(exponent);
}

__kernel
void bilateral(__global outType *        d_dst,
               KParam                    oInfo,
               __global const inType *   d_src,
               KParam                    iInfo,
               __local outType *         localMem,
               __local outType *         gauss2d,
               float sigma_space, float sigma_color,
               dim_type gaussOff, dim_type nonBatchBlkSize)
{
    const dim_type radius      = max((int)(sigma_space * 1.5f), 1);
    const dim_type padding     = 2 * radius;
    const dim_type window_size = padding + 1;
    const dim_type shrdLen     = get_local_size(0) + padding;
    const float variance_range = sigma_color * sigma_color;
    const float variance_space = sigma_space * sigma_space;

    // gfor batch offsets
    unsigned batchId = get_group_id(0) / nonBatchBlkSize;
    __global const inType* in = d_src + (batchId * iInfo.strides[2] + iInfo.offset);
    __global outType* out     = d_dst + (batchId * oInfo.strides[2]);

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

    // generate gauss2d spatial variance values for block
    if (lx<window_size && ly<window_size) {
        int x = lx - radius;
        int y = ly - radius;
        gauss2d[ly*window_size+lx] = exp( ((x*x) + (y*y)) / (-2.f * variance_space));
    }

    // pull image to local memory
    load2LocalMem(localMem, in, lx, ly, shrdLen,
                 iInfo.dims[0], iInfo.dims[1], gx-radius,
                 gy-radius, iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        load2LocalMem(localMem, in, lx2, ly, shrdLen,
                     iInfo.dims[0], iInfo.dims[1], gx2-radius,
                     gy-radius, iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        load2LocalMem(localMem, in, lx, ly2, shrdLen,
                     iInfo.dims[0], iInfo.dims[1], gx-radius,
                     gy2-radius, iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2LocalMem(localMem, in, lx2, ly2, shrdLen,
                     iInfo.dims[0], iInfo.dims[1], gx2-radius,
                     gy2-radius, iInfo.strides[1], iInfo.strides[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1]) {
        outType center_color = localMem[j*shrdLen+i];
        outType res  = 0;
        outType norm = 0;
#pragma unroll
        for(dim_type wj=0; wj<window_size; ++wj) {
            dim_type joff = (j+wj-radius)*shrdLen;
            dim_type goff = wj*window_size;
#pragma unroll
            for(dim_type wi=0; wi<window_size; ++wi) {
                outType tmp_color   = localMem[joff+i+wi-radius];
                outType gauss_space = gauss2d[goff+wi];
                outType gauss_range = gaussian1d(center_color - tmp_color, variance_range);
                outType weight      = gauss_space * gauss_range;
                norm += weight;
                res  += tmp_color * weight;
            }
        }
        dim_type oIdx = gy*oInfo.strides[1] + gx*oInfo.strides[0];
        out[oIdx] = res / norm;
    }
}
