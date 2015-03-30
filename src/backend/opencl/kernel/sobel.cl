/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

Ti load2LocalMem(global const Ti * in,
               dim_type dim0, dim_type dim1,
               dim_type gx, dim_type gy,
               dim_type inStride1, dim_type inStride0)
{
    if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
        return (Ti)0;
    else
        return in[gx*inStride0+gy*inStride1];
}

kernel
void sobel3x3(global To * dx, KParam dxInfo,
              global To * dy, KParam dyInfo,
              global const Ti * in, KParam iInfo,
              local        Ti * localMem,
              dim_type nBBS0, dim_type nBBS1)
{
    const dim_type radius  = 1;
    const dim_type padding = 2*radius;
    const dim_type shrdLen = get_local_size(0) + padding;

    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    global const Ti* iptr = in + (b2 * iInfo.strides[2]  + b3 * iInfo.strides[3] + iInfo.offset);
    global To*      dxptr = dx + (b2 * dxInfo.strides[2] + b3 * dxInfo.strides[3]);
    global To*      dyptr = dy + (b2 * dyInfo.strides[2] + b3 * dyInfo.strides[3]);

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);

    dim_type gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    dim_type gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    dim_type lx2 = lx + get_local_size(0);
    dim_type ly2 = ly + get_local_size(1);
    dim_type gx2 = gx + get_local_size(0);
    dim_type gy2 = gy + get_local_size(1);

    localMem[lx+shrdLen*ly] = load2LocalMem(iptr, iInfo.dims[0], iInfo.dims[1],
                                   gx-radius, gy-radius,
                                   iInfo.strides[1], iInfo.strides[0]);
    if (lx<padding) {
        localMem[lx2+shrdLen*ly] = load2LocalMem(iptr, iInfo.dims[0], iInfo.dims[1],
                                        gx2-radius, gy-radius,
                                        iInfo.strides[1], iInfo.strides[0]);
    }
    if (ly<padding) {
        localMem[lx+shrdLen*ly2] = load2LocalMem(iptr, iInfo.dims[0], iInfo.dims[1],
                                        gx-radius, gy2-radius,
                                        iInfo.strides[1], iInfo.strides[0]);
    }
    if (lx<padding && ly<padding) {
        localMem[lx2+shrdLen*ly2] = load2LocalMem(iptr, iInfo.dims[0], iInfo.dims[1],
                                         gx2-radius, gy2-radius,
                                         iInfo.strides[1], iInfo.strides[0]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < iInfo.dims[0] && gy < iInfo.dims[1]) {
        dim_type i = lx + radius;
        dim_type j = ly + radius;
        dim_type _i = i-1;
        dim_type i_ = i+1;
        dim_type _j = j-1;
        dim_type j_ = j+1;

        float NW = localMem[_i+shrdLen*_j];
        float SW = localMem[i_+shrdLen*_j];
        float NE = localMem[_i+shrdLen*j_];
        float SE = localMem[i_+shrdLen*j_];

        float t1 = localMem[i+shrdLen*_j];
        float t2 = localMem[i+shrdLen*j_];
        dxptr[gy*dxInfo.strides[1]+gx] = (NW+SW - (NE+SE) + 2*(t1-t2));

        t1 = localMem[_i+shrdLen*j];
        t2 = localMem[i_+shrdLen*j];
        dyptr[gy*dyInfo.strides[1]+gx] = (NW+NE - (SW+SE) + 2*(t1-t2));

    }
}
