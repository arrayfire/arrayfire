/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Exchange trick: Morgan McGuire, ShaderX 2008
#define swap(a,b)    { T tmp = a; a = min(a,b); b = max(tmp,b); }

dim_type lIdx(dim_type x, dim_type y, dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

void load2ShrdMem(__local T *           shrd,
                  __global const T *    in,
                  dim_type lx, dim_type ly, dim_type shrdStride,
                  dim_type dim0, dim_type dim1,
                  dim_type gx, dim_type gy,
                  dim_type inStride1, dim_type inStride0)
{
    if (pad==AF_ZERO) {
        if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
            shrd[lIdx(lx, ly, shrdStride, 1)] = (T)0;
        else
            shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
    } else if (pad==AF_SYMMETRIC) {
        if (gx<0) gx *= -1;
        if (gy<0) gy *= -1;
        if (gx>=dim0) gx = 2*(dim0-1) - gx;
        if (gy>=dim1) gy = 2*(dim1-1) - gy;
        shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx, gy, inStride1, inStride0)];
    }
}

__kernel
void medfilt(__global T *       out,
             KParam             oInfo,
             __global const T * in,
             KParam             iInfo,
             __local T *        localMem,
             dim_type           nBBS0,
             dim_type           nBBS1)
{
    // calculate necessary offset and window parameters
    const dim_type padding = w_len-1;
    const dim_type halo    = padding/2;
    const dim_type shrdLen = get_local_size(0) + padding;

    // batch offsets
    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    __global const T* iptr =  in + (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    __global T*       optr = out + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]);

    // local neighborhood indices
    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);

    // global indices
    dim_type gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    dim_type gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    // offset values for pulling image to local memory
    dim_type lx2 = lx + get_local_size(0);
    dim_type ly2 = ly + get_local_size(1);
    dim_type gx2 = gx + get_local_size(0);
    dim_type gy2 = gy + get_local_size(1);

    // pull image to local memory
    load2ShrdMem(localMem, iptr, lx, ly, shrdLen,
                 iInfo.dims[0], iInfo.dims[1],
                 gx-halo, gy-halo,
                 iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        load2ShrdMem(localMem, iptr, lx2, ly, shrdLen,
                     iInfo.dims[0], iInfo.dims[1],
                     gx2-halo, gy-halo,
                     iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem(localMem, iptr, lx, ly2, shrdLen,
                     iInfo.dims[0], iInfo.dims[1],
                     gx-halo, gy2-halo,
                     iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem(localMem, iptr, lx2, ly2, shrdLen,
                     iInfo.dims[0], iInfo.dims[1],
                     gx2-halo, gy2-halo,
                     iInfo.strides[1], iInfo.strides[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Only continue if we're at a valid location
    if (gx < iInfo.dims[0] && gy < iInfo.dims[1]) {

        // pull top half from shared memory into local memory
        T v[ARR_SIZE];
#pragma unroll
        for(dim_type k = 0; k <= w_wid/2; k++) {
#pragma unroll
            for(dim_type i = 0; i < w_len; i++) {
                v[w_len*k + i] = localMem[lIdx(lx+i,ly+k,shrdLen,1)];
            }
        }

        // with each pass, remove min and max values and add new value
        // initial sort
        // ensure min in first half, max in second half
#pragma unroll
        for(dim_type i = 0; i < ARR_SIZE/2; i++) {
            swap(v[i], v[ARR_SIZE-1-i]);
        }
        // move min in first half to first pos
#pragma unroll
        for(dim_type i = 1; i < (ARR_SIZE+1)/2; i++) {
            swap(v[0], v[i]);
        }
        // move max in second half to last pos
#pragma unroll
        for(dim_type i = ARR_SIZE-2; i >= ARR_SIZE/2; i--) {
            swap(v[i], v[ARR_SIZE-1]);
        }

        dim_type last = ARR_SIZE-1;

        for(dim_type k = 1+w_wid/2; k < w_wid; k++) {

            for(dim_type j = 0; j < w_len; j++) {

                // add new contestant to first position in array
                v[0] = localMem[lIdx(lx+j, ly+k, shrdLen, 1)];

                last--;

                // place max in last half, min in first half
                for(dim_type i = 0; i < (last+1)/2; i++) {
                    swap(v[i], v[last-i]);
                }
                // now perform swaps on each half such that
                // max is in last pos, min is in first pos
                for(dim_type i = 1; i <= last/2; i++) {
                    swap(v[0], v[i]);
                }
                for(dim_type i = last-1; i >= (last+1)/2; i--) {
                    swap(v[i], v[last]);
                }
            }
        }

        // no more new contestants
        // may still have to sort the last row
        // each outer loop drops the min and max
        for(dim_type k = 1; k < w_len/2; k++) {
            // move max/min into respective halves
            for(dim_type i = k; i < w_len/2; i++) {
                swap(v[i], v[w_len-1-i]);
            }
            // move min into first pos
            for(dim_type i = k+1; i <= w_len/2; i++) {
                swap(v[k], v[i]);
            }
            // move max into last pos
            for(dim_type i = w_len-k-2; i >= w_len/2; i--) {
                swap(v[i], v[w_len-1-k]);
            }
        }

        // pick the middle element of the first row
        optr[gy*oInfo.strides[1]+gx*oInfo.strides[0]] = v[w_len/2];
    }
}
