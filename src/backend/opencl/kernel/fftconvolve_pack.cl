/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void pack_data(
    __global CONVT   *d_out,
    KParam            oInfo,
    __global const T *d_in,
    KParam            iInfo,
    const dim_type    di0_half,
    const int         odd_di0)
{
    const int t = get_global_id(0);

    const int tMax = oInfo.strides[3] * oInfo.dims[3];

    if (t >= tMax)
        return;

    const dim_type do0 = oInfo.dims[0];
    const dim_type do1 = oInfo.dims[1];
    const dim_type do2 = oInfo.dims[2];

    const dim_type so1 = oInfo.strides[1];
    const dim_type so2 = oInfo.strides[2];
    const dim_type so3 = oInfo.strides[3];

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = t / so3;

    const dim_type di0 = iInfo.dims[0];
    const dim_type di1 = iInfo.dims[1];
    const dim_type di2 = iInfo.dims[2];

    const dim_type si1 = iInfo.strides[1];
    const dim_type si2 = iInfo.strides[2];
    const dim_type si3 = iInfo.strides[3];

    const int ti0 = to0;
    const int ti1 = to1 * si1;
    const int ti2 = to2 * si2;
    const int ti3 = to3 * si3;

    const int iidx1 = iInfo.offset + ti3 + ti2 + ti1 + ti0;
    const int iidx2 = iidx1 + di0_half;

    // Treating complex output array as real-only array,
    // thus, multiply strides by 2
    const int oidx1 = oInfo.offset + to3*so3*2 + to2*so2*2 + to1*so1*2 + to0*2;
    const int oidx2 = oidx1 + 1;

    if (to0 < di0_half && to1 < di1 && to2 < di2) {
        d_out[oidx1] = (CONVT)d_in[iidx1];
        if (ti0 == di0_half-1 && odd_di0 == 1)
            d_out[oidx2] = (CONVT)0;
        else
            d_out[oidx2] = (CONVT)d_in[iidx2];
    }
    else {
        // Pad remaining elements with 0s
        d_out[oidx1] = (CONVT)0;
        d_out[oidx2] = (CONVT)0;
    }
}

__kernel
void pad_array(
    __global CONVT   *d_out,
    KParam            oInfo,
    __global const T *d_in,
    KParam            iInfo)
{
    const int t = get_global_id(0);

    const int tMax = oInfo.strides[3] * oInfo.dims[3];

    if (t >= tMax)
        return;

    const dim_type do0 = oInfo.dims[0];
    const dim_type do1 = oInfo.dims[1];
    const dim_type do2 = oInfo.dims[2];

    const dim_type so1 = oInfo.strides[1];
    const dim_type so2 = oInfo.strides[2];
    const dim_type so3 = oInfo.strides[3];

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = (t / so3);

    const dim_type di0 = iInfo.dims[0];
    const dim_type di1 = iInfo.dims[1];
    const dim_type di2 = iInfo.dims[2];
    const dim_type di3 = iInfo.dims[3];

    const dim_type si1 = iInfo.strides[1];
    const dim_type si2 = iInfo.strides[2];
    const dim_type si3 = iInfo.strides[3];

    const int ti0 = to0;
    const int ti1 = to1 * si1;
    const int ti2 = to2 * si2;
    const int ti3 = to3 * si3;

    const int iidx = iInfo.offset + ti3 + ti2 + ti1 + ti0;

    const int oidx = oInfo.offset + t*2;

    if (to0 < di0 && to1 < di1 && to2 < di2 && to3 < di3) {
        // Copy input elements to real elements, set imaginary elements to 0
        d_out[oidx]   = (CONVT)d_in[iidx];
        d_out[oidx+1] = (CONVT)0;
    }
    else {
        // Pad remaining of the matrix to 0s
        d_out[oidx]   = (CONVT)0;
        d_out[oidx+1] = (CONVT)0;
    }
}
