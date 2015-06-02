/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

__kernel
void reorder_output(
    __global T           *d_out,
    KParam                oInfo,
    __global const CONVT *d_in,
    KParam                iInfo,
    KParam                fInfo,
    const int        half_di0,
    const int             baseDim)
{
    const int t = get_global_id(0);

    const int tMax = oInfo.strides[3] * oInfo.dims[3];

    if (t >= tMax)
        return;

    const int do0 = oInfo.dims[0];
    const int do1 = oInfo.dims[1];
    const int do2 = oInfo.dims[2];

    const int so1 = oInfo.strides[1];
    const int so2 = oInfo.strides[2];
    const int so3 = oInfo.strides[3];

    // Treating complex input array as real-only array,
    // thus, multiply dimension 0 and strides by 2
    const int di0 = iInfo.dims[0] * 2;
    const int di1 = iInfo.dims[1];
    const int di2 = iInfo.dims[2];

    const int si1 = iInfo.strides[1] * 2;
    const int si2 = iInfo.strides[2] * 2;
    const int si3 = iInfo.strides[3] * 2;

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = (t / so3);

    int oidx = to3*so3 + to2*so2 + to1*so1 + to0;

    int ti0, ti1, ti2, ti3;
#if EXPAND == 1
    ti0 = to0;
    ti1 = to1 * si1;
    ti2 = to2 * si2;
    ti3 = to3 * si3;
#else
    ti0 = to0 + fInfo.dims[0]/2;
    ti1 = (to1 + (baseDim > 1)*(fInfo.dims[1]/2)) * si1;
    ti2 = (to2 + (baseDim > 2)*(fInfo.dims[2]/2)) * si2;
    ti3 = to3 * si3;
#endif

    // Divide output elements to cuFFT resulting scale, round result if output
    // type is single or double precision floating-point
    if (ti0 < half_di0) {
        // Copy top elements
        int iidx = iInfo.offset + ti3 + ti2 + ti1 + ti0 * 2;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round(d_in[iidx]);
#else
            d_out[oidx] = (T)(d_in[iidx]);
#endif
    }
    else if (ti0 < half_di0 + fInfo.dims[0] - 1) {
        // Add central elements
        int iidx1 = iInfo.offset + ti3 + ti2 + ti1 + ti0 * 2;
        int iidx2 = iInfo.offset + ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round((d_in[iidx1] + d_in[iidx2]));
#else
            d_out[oidx] = (T)((d_in[iidx1] + d_in[iidx2]));
#endif
    }
    else {
        // Copy bottom elements
        const int iidx = iInfo.offset + ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round(d_in[iidx]);
#else
            d_out[oidx] = (T)(d_in[iidx]);
#endif
    }
}
