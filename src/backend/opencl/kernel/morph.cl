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
        __global const T *      in,
        int lx, int ly, int shrdStride,
        int dim0, int dim1,
        int gx, int gy,
        int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

//kernel assumes four dimensions
//doing this to reduce one uneccesary parameter
__kernel
void morph(__global T *              out,
           KParam                  oInfo,
           __global const T *         in,
           KParam                  iInfo,
           __constant const T *   d_filt,
           __local T *          localMem,
           int nBBS0, int nBBS1)
{
    const int halo   = windLen/2;
    const int padding= 2*halo;
    const int shrdLen= get_local_size(0) + padding + 1;

    // gfor batch offsets
    int b2 = get_group_id(0) / nBBS0;
    int b3 = get_group_id(1) / nBBS1;
    in  += (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    out += (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    // local neighborhood indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // global indices
    int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    // offset values for pulling image to local memory
    int lx2      = lx + get_local_size(0);
    int ly2      = ly + get_local_size(1);
    int gx2      = gx + get_local_size(0);
    int gy2      = gy + get_local_size(1);

    // pull image to local memory
    load2LocalMem(localMem, in, lx, ly, shrdLen,
                  iInfo.dims[0], iInfo.dims[1],
                  gx-halo, gy-halo,
                  iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        load2LocalMem(localMem, in, lx2, ly, shrdLen,
                      iInfo.dims[0], iInfo.dims[1],
                      gx2-halo, gy-halo,
                      iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        load2LocalMem(localMem, in, lx, ly2, shrdLen,
                      iInfo.dims[0], iInfo.dims[1],
                      gx-halo, gy2-halo,
                      iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2LocalMem(localMem, in, lx2, ly2, shrdLen,
                      iInfo.dims[0], iInfo.dims[1],
                      gx2-halo, gy2-halo,
                      iInfo.strides[1], iInfo.strides[0]);
    }

    int i = lx + halo;
    int j = ly + halo;
    barrier(CLK_LOCAL_MEM_FENCE);

    T acc = localMem[ lIdx(i, j, shrdLen, 1) ];
#pragma unroll
    for(int wj=0; wj<windLen; ++wj) {
        int joff   = wj*windLen;
        int w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
        for(int wi=0; wi<windLen; ++wi) {
            T cur  = localMem[w_joff+i+wi-halo];
            if (d_filt[joff+wi]) {
                if (isDilation)
                    acc = max(acc, cur);
                else
                    acc = min(acc, cur);
            }
        }
    }

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1]) {
        int outIdx = lIdx(gx, gy, oInfo.strides[1], oInfo.strides[0]);
        out[outIdx] = acc;
    }
}



int lIdx3D(int x, int y, int z,
        int stride2, int stride1, int stride0)
{
    return (z*stride2 + y*stride1 + x*stride0);
}

void load2LocVolume(__local T * shrd,
        __global const T * in,
        int lx, int ly, int lz,
        int shrdStride1, int shrdStride2,
        int dim0, int dim1, int dim2,
        int gx, int gy, int gz,
        int inStride2, int inStride1, int inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
    int gz_  = clamp(gz, 0, dim2-1);
    int shrdIdx = lx + ly*shrdStride1 + lz*shrdStride2;
    int inIdx   = gx_*inStride0 + gy_*inStride1 + gz_*inStride2;
    shrd[ shrdIdx ] = in[ inIdx ];
}

__kernel
void morph3d(__global T *         out,
             KParam               oInfo,
             __global const T *   in,
             KParam               iInfo,
             __constant const T * d_filt,
             __local T *          localMem,
             int             nBBS)
{
    const int halo   = windLen/2;
    const int padding= 2*halo;

    const int se_area   = windLen*windLen;
    const int shrdLen   = get_local_size(0) + padding + 1;
    const int shrdArea  = shrdLen * (get_local_size(1)+padding);

    // gfor batch offsets
    int batchId    = get_group_id(0) / nBBS;
    in  += (batchId * iInfo.strides[3] + iInfo.offset);
    out += (batchId * oInfo.strides[3]);

    int gx, gy, gz, i, j, k;
    { // scoping out unnecessary variables
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lz = get_local_id(2);

    gx = get_local_size(0) * (get_group_id(0)-batchId*nBBS) + lx;
    gy = get_local_size(1) * get_group_id(1) + ly;
    gz = get_local_size(2) * get_group_id(2) + lz;

    const int gx2 = gx + get_local_size(0);
    const int gy2 = gy + get_local_size(1);
    const int gz2 = gz + get_local_size(2);
    const int lx2 = lx + get_local_size(0);
    const int ly2 = ly + get_local_size(1);
    const int lz2 = lz + get_local_size(2);

    // pull volume to shared memory
    load2LocVolume(localMem, in, lx, ly, lz, shrdLen, shrdArea,
                    iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                    gx-halo, gy-halo, gz-halo,
                    iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        load2LocVolume(localMem, in, lx2, ly, lz, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx2-halo, gy-halo, gz-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        load2LocVolume(localMem, in, lx, ly2, lz, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx-halo, gy2-halo, gz-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (lz<padding) {
        load2LocVolume(localMem, in, lx, ly, lz2, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx-halo, gy-halo, gz2-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2LocVolume(localMem, in, lx2, ly2, lz, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx2-halo, gy2-halo, gz-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding && lz<padding) {
        load2LocVolume(localMem, in, lx, ly2, lz2, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx-halo, gy2-halo, gz2-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (lz<padding && lx<padding) {
        load2LocVolume(localMem, in, lx2, ly, lz2, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx2-halo, gy-halo, gz2-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding && lz<padding) {
        load2LocVolume(localMem, in, lx2, ly2, lz2, shrdLen, shrdArea,
                       iInfo.dims[0], iInfo.dims[1], iInfo.dims[2],
                       gx2-halo, gy2-halo, gz2-halo,
                       iInfo.strides[2], iInfo.strides[1], iInfo.strides[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // indices of voxel owned by current thread
    i  = lx + halo;
    j  = ly + halo;
    k  = lz + halo;
    }

    T acc = localMem[ lIdx3D(i, j, k, shrdArea, shrdLen, 1) ];
#pragma unroll
    for(int wk=0; wk<windLen; ++wk) {
        int koff   = wk*se_area;
        int w_koff = (k+wk-halo)*shrdArea;
#pragma unroll
        for(int wj=0; wj<windLen; ++wj) {
        int joff   = wj*windLen;
        int w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
            for(int wi=0; wi<windLen; ++wi) {
                T cur  = localMem[w_koff+w_joff + i+wi-halo];
                if (d_filt[koff+joff+wi]) {
                    if (isDilation)
                        acc = max(acc, cur);
                    else
                        acc = min(acc, cur);
                }
            }
        }
    }

    if (gx<oInfo.dims[0] && gy<oInfo.dims[1] && gz<oInfo.dims[2]) {
        int outIdx = gz * oInfo.strides[2] +
            gy * oInfo.strides[1] +
            gx * oInfo.strides[0];
        out[outIdx] = acc;
    }
}
