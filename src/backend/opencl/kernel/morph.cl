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
        __global const T *      in,
        dim_type lx, dim_type ly, dim_type shrdStride,
        dim_type dim0, dim_type dim1,
        dim_type gx, dim_type gy,
        dim_type inStride1, dim_type inStride0)
{
    dim_type gx_  = clamp(gx, 0, dim0-1);
    dim_type gy_  = clamp(gy, 0, dim1-1);
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
           dim_type nBBS0, dim_type nBBS1)
{
    const dim_type halo   = windLen/2;
    const dim_type padding= 2*halo;
    const dim_type shrdLen= get_local_size(0) + padding + 1;

    // gfor batch offsets
    dim_type b2 = get_group_id(0) / nBBS0;
    dim_type b3 = get_group_id(1) / nBBS1;
    in  += (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    out += (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    // local neighborhood indices
    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    // global indices
    dim_type gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    dim_type gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    // offset values for pulling image to local memory
    dim_type lx2      = lx + get_local_size(0);
    dim_type ly2      = ly + get_local_size(1);
    dim_type gx2      = gx + get_local_size(0);
    dim_type gy2      = gy + get_local_size(1);

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

    dim_type i = lx + halo;
    dim_type j = ly + halo;
    barrier(CLK_LOCAL_MEM_FENCE);

    T acc = localMem[ lIdx(i, j, shrdLen, 1) ];
#pragma unroll
    for(dim_type wj=0; wj<windLen; ++wj) {
        dim_type joff   = wj*windLen;
        dim_type w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
        for(dim_type wi=0; wi<windLen; ++wi) {
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
        dim_type outIdx = lIdx(gx, gy, oInfo.strides[1], oInfo.strides[0]);
        out[outIdx] = acc;
    }
}



dim_type lIdx3D(dim_type x, dim_type y, dim_type z,
        dim_type stride2, dim_type stride1, dim_type stride0)
{
    return (z*stride2 + y*stride1 + x*stride0);
}

void load2LocVolume(__local T * shrd,
        __global const T * in,
        dim_type lx, dim_type ly, dim_type lz,
        dim_type shrdStride1, dim_type shrdStride2,
        dim_type dim0, dim_type dim1, dim_type dim2,
        dim_type gx, dim_type gy, dim_type gz,
        dim_type inStride2, dim_type inStride1, dim_type inStride0)
{
    dim_type gx_  = clamp(gx, 0, dim0-1);
    dim_type gy_  = clamp(gy, 0, dim1-1);
    dim_type gz_  = clamp(gz, 0, dim2-1);
    dim_type shrdIdx = lx + ly*shrdStride1 + lz*shrdStride2;
    dim_type inIdx   = gx_*inStride0 + gy_*inStride1 + gz_*inStride2;
    shrd[ shrdIdx ] = in[ inIdx ];
}

__kernel
void morph3d(__global T *         out,
             KParam               oInfo,
             __global const T *   in,
             KParam               iInfo,
             __constant const T * d_filt,
             __local T *          localMem,
             dim_type             nBBS)
{
    const dim_type halo   = windLen/2;
    const dim_type padding= 2*halo;

    const dim_type se_area   = windLen*windLen;
    const dim_type shrdLen   = get_local_size(0) + padding + 1;
    const dim_type shrdArea  = shrdLen * (get_local_size(1)+padding);

    // gfor batch offsets
    dim_type batchId    = get_group_id(0) / nBBS;
    in  += (batchId * iInfo.strides[3] + iInfo.offset);
    out += (batchId * oInfo.strides[3]);

    dim_type gx, gy, gz, i, j, k;
    { // scoping out unnecessary variables
    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);
    const dim_type lz = get_local_id(2);

    gx = get_local_size(0) * (get_group_id(0)-batchId*nBBS) + lx;
    gy = get_local_size(1) * get_group_id(1) + ly;
    gz = get_local_size(2) * get_group_id(2) + lz;

    const dim_type gx2 = gx + get_local_size(0);
    const dim_type gy2 = gy + get_local_size(1);
    const dim_type gz2 = gz + get_local_size(2);
    const dim_type lx2 = lx + get_local_size(0);
    const dim_type ly2 = ly + get_local_size(1);
    const dim_type lz2 = lz + get_local_size(2);

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
    for(dim_type wk=0; wk<windLen; ++wk) {
        dim_type koff   = wk*se_area;
        dim_type w_koff = (k+wk-halo)*shrdArea;
#pragma unroll
        for(dim_type wj=0; wj<windLen; ++wj) {
        dim_type joff   = wj*windLen;
        dim_type w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
            for(dim_type wi=0; wi<windLen; ++wi) {
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
        dim_type outIdx = gz * oInfo.strides[2] +
                          gy * oInfo.strides[1] +
                          gx * oInfo.strides[0];
        out[outIdx] = acc;
    }
}
