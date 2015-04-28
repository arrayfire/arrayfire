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

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;
    const dim_type do0_half = do0/2;
    const dim_type do01_half = do0_half * do1;
    const dim_type do012_half = do01_half * do2;

    const int to0 = t % do0_half;
    const int to1 = (t / do0_half) % do1;
    const int to2 = (t / do01_half) % do2;
    const int to3 = t / do012_half;

    const dim_type di0 = iInfo.dims[0];
    const dim_type di1 = iInfo.dims[1];
    const dim_type di2 = iInfo.dims[2];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int ti0 = to0;
    const int ti1 = to1 * di0;
    const int ti2 = to2 * di01;
    const int ti3 = to3 * di012;

    const int iidx1 = ti3 + ti2 + ti1 + ti0;
    const int iidx2 = iidx1 + di0_half;
    const int oidx1 = to3*do012 + to2*do01 + to1*do0 + to0*2;
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
    const dim_type    oOff,
    __global const T *d_in,
    KParam            iInfo)
{
    const int t = get_global_id(0);

    const int tMax = oInfo.strides[3] * oInfo.dims[3];

    if (t >= tMax)
        return;

    const dim_type do0 = oInfo.dims[0]/2;
    const dim_type do1 = oInfo.dims[1];
    const dim_type do2 = oInfo.dims[2];

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;

    const int to0 = t % do0;
    const int to1 = (t / do0) % do1;
    const int to2 = (t / do01) % do2;
    const int to3 = (t / do012);

    const dim_type di0 = iInfo.dims[0];
    const dim_type di1 = iInfo.dims[1];
    const dim_type di2 = iInfo.dims[2];
    const dim_type di3 = iInfo.dims[3];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int ti0 = to0;
    const int ti1 = to1 * di0;
    const int ti2 = to2 * di01;
    const int ti3 = to3 * di012;

    const int iidx = ti3 + ti2 + ti1 + ti0;

    const int oidx = oOff + t*2;

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

__kernel
void complex_multiply(
    __global CONVT       *d_out,
    KParam                oInfo,
    const dim_type        oOff,
    __global const CONVT *d_in1,
    KParam                i1Info,
    const dim_type        i1Off,
    __global const CONVT *d_in2,
    KParam                i2Info,
    const dim_type        i2Off,
    const dim_type        nelem,
    const int             kind)
{
    const int t = get_global_id(0);

    if (t >= nelem)
        return;

    if (kind == ONE2ONE || kind == MANY2MANY) {
        // Complex multiply each signal to equivalent filter
        const int ridx = t * 2;
        const int iidx = t * 2 + 1;

        CONVT a = d_in1[i1Off + ridx];
        CONVT b = d_in1[i1Off + iidx];
        CONVT c = d_in2[i2Off + ridx];
        CONVT d = d_in2[i2Off + iidx];

        CONVT ac = a*c;
        CONVT bd = b*d;

        d_out[oOff + ridx] = ac - bd;
        d_out[oOff + iidx] = (a+b) * (c+d) - ac - bd;
    }
    else if (kind == MANY2ONE) {
        // Complex multiply all signals to filter
        const int ridx1 = t * 2;
        const int iidx1 = t * 2 + 1;
        const int ridx2 = (t*2)   % (i2Info.strides[3] * i2Info.dims[3]);
        const int iidx2 = (t*2+1) % (i2Info.strides[3] * i2Info.dims[3]);

        CONVT a = d_in1[i1Off + ridx1];
        CONVT b = d_in1[i1Off + iidx1];
        CONVT c = d_in2[i2Off + ridx2];
        CONVT d = d_in2[i2Off + iidx2];

        CONVT ac = a*c;
        CONVT bd = b*d;

        d_out[oOff + ridx1] = ac - bd;
        d_out[oOff + iidx1] = (a+b) * (c+d) - ac - bd;
    }
    else if (kind == ONE2MANY) {
        // Complex multiply signal to all filters
        const int ridx1 = (t*2)   % (i1Info.strides[3] * i1Info.dims[3]);
        const int iidx1 = (t*2+1) % (i1Info.strides[3] * i1Info.dims[3]);
        const int ridx2 = t * 2;
        const int iidx2 = t * 2 + 1;

        CONVT a = d_in1[i1Off + ridx1];
        CONVT b = d_in1[i1Off + iidx1];
        CONVT c = d_in2[i2Off + ridx2];
        CONVT d = d_in2[i2Off + iidx2];

        CONVT ac = a*c;
        CONVT bd = b*d;

        d_out[oOff + ridx2] = ac - bd;
        d_out[oOff + iidx2] = (a+b) * (c+d) - ac - bd;
    }
}

__kernel
void reorder_output(
    __global T           *d_out,
    KParam                oInfo,
    __global const CONVT *d_in,
    KParam                iInfo,
    const dim_type        iOff,
    KParam                fInfo,
    const dim_type        half_di0,
    const int             baseDim,
    const int             fftScale)
{
    const int t = get_global_id(0);

    const int tMax = oInfo.strides[3] * oInfo.dims[3];

    if (t >= tMax)
        return;

    const dim_type do0 = oInfo.dims[0];
    const dim_type do1 = oInfo.dims[1];
    const dim_type do2 = oInfo.dims[2];

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;

    const dim_type di0 = iInfo.dims[0];
    const dim_type di1 = iInfo.dims[1];
    const dim_type di2 = iInfo.dims[2];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int to0 = t % do0;
    const int to1 = (t / do0) % do1;
    const int to2 = (t / do01) % do2;
    const int to3 = (t / do012);

    int oidx = to3*do012 + to2*do01 + to1*do0 + to0;

    int ti0, ti1, ti2, ti3;
#if EXPAND == 1
        ti0 = to0;
        ti1 = to1 * di0;
        ti2 = to2 * di01;
        ti3 = to3 * di012;
#else
        ti0 = to0 + fInfo.dims[0]/2;
        ti1 = (to1 + (baseDim > 1)*(fInfo.dims[1]/2)) * di0;
        ti2 = (to2 + (baseDim > 2)*(fInfo.dims[2]/2)) * di01;
        ti3 = to3 * di012;
#endif

    // Divide output elements to cuFFT resulting scale, round result if output
    // type is single or double precision floating-point
    if (ti0 < half_di0) {
        // Copy top elements
        int iidx = iOff + ti3 + ti2 + ti1 + ti0 * 2;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round(d_in[iidx]);
#else
            d_out[oidx] = (T)(d_in[iidx]);
#endif
    }
    else if (ti0 < half_di0 + fInfo.dims[0] - 1) {
        // Add central elements
        int iidx1 = iOff + ti3 + ti2 + ti1 + ti0 * 2;
        int iidx2 = iOff + ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round((d_in[iidx1] + d_in[iidx2]));
#else
            d_out[oidx] = (T)((d_in[iidx1] + d_in[iidx2]));
#endif
    }
    else {
        // Copy bottom elements
        const int iidx = iOff + ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
#if ROUND_OUT == 1
            d_out[oidx] = (T)round(d_in[iidx]);
#else
            d_out[oidx] = (T)(d_in[iidx]);
#endif
    }
}
