/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

int reflect101(int index, int endIndex) {
    return abs(endIndex - (int)abs(endIndex - index));
}

Ti load2LocalMem(global const Ti* in, int d0, int d1, int gx, int gy,
                 int inStride1, int inStride0) {
    int idx =
        reflect101(gx, d0 - 1) * inStride0 + reflect101(gy, d1 - 1) * inStride1;
    return in[idx];
}

kernel void sobel3x3(global To* dx, KParam dxInfo, global To* dy, KParam dyInfo,
                     global const Ti* in, KParam iInfo, local Ti* localMem,
                     int nBBS0, int nBBS1) {
    const int radius  = 1;
    const int padding = 2 * radius;
    const int shrdLen = get_local_size(0) + padding;

    unsigned b2 = get_group_id(0) / nBBS0;
    unsigned b3 = get_group_id(1) / nBBS1;
    global const Ti* iptr =
        in + (b2 * iInfo.strides[2] + b3 * iInfo.strides[3] + iInfo.offset);
    global To* dxptr = dx + (b2 * dxInfo.strides[2] + b3 * dxInfo.strides[3]);
    global To* dyptr = dy + (b2 * dyInfo.strides[2] + b3 * dyInfo.strides[3]);

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gx = get_local_size(0) * (get_group_id(0) - b2 * nBBS0) + lx;
    int gy = get_local_size(1) * (get_group_id(1) - b3 * nBBS1) + ly;

    int s0 = iInfo.strides[0];
    int s1 = iInfo.strides[1];
    int d0 = iInfo.dims[0];
    int d1 = iInfo.dims[1];
    for (int b = ly, gy2 = gy; b < shrdLen;
         b += get_local_size(1), gy2 += get_local_size(1)) {
        for (int a = lx, gx2 = gx; a < shrdLen;
             a += get_local_size(0), gx2 += get_local_size(0)) {
            localMem[a + shrdLen * b] =
                load2LocalMem(iptr, d0, d1, gx2 - radius, gy2 - radius, s1, s0);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < iInfo.dims[0] && gy < iInfo.dims[1]) {
        int i  = lx + radius;
        int j  = ly + radius;
        int _i = i - 1;
        int i_ = i + 1;
        int _j = j - 1;
        int j_ = j + 1;

        float NW = localMem[_i + shrdLen * _j];
        float SW = localMem[i_ + shrdLen * _j];
        float NE = localMem[_i + shrdLen * j_];
        float SE = localMem[i_ + shrdLen * j_];

        float t1 = localMem[_i + shrdLen * j];
        float t2 = localMem[i_ + shrdLen * j];
        dxptr[gy * dxInfo.strides[1] + gx] =
            (SW + SE - (NW + NE) + 2 * (t2 - t1));

        t1 = localMem[i + shrdLen * _j];
        t2 = localMem[i + shrdLen * j_];
        dyptr[gy * dyInfo.strides[1] + gx] =
            (NE + SE - (NW + SW) + 2 * (t2 - t1));
    }
}
