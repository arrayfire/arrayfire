/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifdef USE_NATIVE_EXP
#define EXP native_exp
#else
#define EXP exp
#endif

int lIdx(int x, int y,
        int stride1, int stride0)
{
    return (y*stride1 + x*stride0);
}

void load2LocalMem(__local outType *  shrd,
        __global const inType *      in,
        int lx, int ly, int shrdStride,
        int dim0, int dim1,
        int gx, int gy,
        int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = (outType)in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

__kernel
void bilateral(__global outType *        d_dst,
               KParam                    oInfo,
               __global const inType *   d_src,
               KParam                    iInfo,
               __local outType *         localMem,
               __local outType *         gauss2d,
               float sigma_space, float sigma_color,
               int gaussOff, int nBBS0, int nBBS1)
{
    const int radius      = max((int)(sigma_space * 1.5f), 1);
    const int padding     = 2 * radius;
    const int window_size = padding + 1;
    const int shrdLen     = get_local_size(0) + padding;
    const float variance_range = sigma_color * sigma_color;
    const float variance_space = sigma_space * sigma_space;
    const float variance_space_neg2 = -2.0 * variance_space;
    const float inv_variance_range_neg2 = -0.5 / (variance_range);

    // gfor batch offsets
    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    __global const inType* in = d_src + (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    __global outType* out     = d_dst + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    // generate gauss2d spatial variance values for block
    if (lx<window_size && ly<window_size) {
        int x = lx - radius;
        int y = ly - radius;
        gauss2d[ly*window_size+lx] = EXP(((x*x) + (y*y)) / variance_space_neg2);
    }

    int s0 = iInfo.strides[0];
    int s1 = iInfo.strides[1];
    int d0 = iInfo.dims[0];
    int d1 = iInfo.dims[1];
    // pull image to local memory
    for (int b=ly, gy2=gy; b<shrdLen; b+=get_local_size(1), gy2+=get_local_size(1)) {
        // move row_set get_local_size(1) along coloumns
        for (int a=lx, gx2=gx; a<shrdLen; a+=get_local_size(0), gx2+=get_local_size(0)) {
            load2LocalMem(localMem, in, a, b, shrdLen, d0, d1, gx2-radius, gy2-radius, s1, s0);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<iInfo.dims[0] && gy<iInfo.dims[1]) {
        lx += radius;
        ly += radius;
        outType center_color = localMem[ly*shrdLen+lx];
        outType res  = 0;
        outType norm = 0;

        int joff = (ly - radius)*shrdLen + (lx-radius);
        int goff = 0;

#pragma unroll
        for(int wj=0; wj<window_size; ++wj) {
#pragma unroll
            for(int wi=0; wi<window_size; ++wi) {
                outType tmp_color   = localMem[joff+wi];
                const outType c = center_color - tmp_color;
                outType gauss_range = EXP(c * c * inv_variance_range_neg2);
                outType weight      = gauss2d[goff+wi] * gauss_range;
                norm += weight;
                res  += tmp_color * weight;
            }
            joff += shrdLen;
            goff += window_size;
        }
        out[gy*oInfo.strides[1] + gx] = res / norm;
    }
}
