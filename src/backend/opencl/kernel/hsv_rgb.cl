/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel
void convert(global T * out, KParam oInfo, global const T * in, KParam iInfo, int nBBS)
{
    // batch offsets
    unsigned batchId = get_group_id(0) / nBBS;
    global const T* src =  in + (batchId * iInfo.strides[3]);
    global T*       dst = out + (batchId * oInfo.strides[3]);
    // global indices
    int gx = get_local_size(0) * (get_group_id(0)-batchId*nBBS) + get_local_id(0);
    int gy = get_local_size(1) * get_group_id(1) + get_local_id(1);

    if (gx < oInfo.dims[0] && gy < oInfo.dims[1]) {

        int oIdx0 = gx + gy * oInfo.strides[1];
        int oIdx1 = oIdx0 + oInfo.strides[2];
        int oIdx2 = oIdx1 + oInfo.strides[2];

        int iIdx0 = gx * iInfo.strides[0] + gy * iInfo.strides[1];
        int iIdx1 = iIdx0 + iInfo.strides[2];
        int iIdx2 = iIdx1 + iInfo.strides[2];

#ifdef isHSV2RGB
        T H = src[iIdx0];
        T S = src[iIdx1];
        T V = src[iIdx2];

        T R, G, B;
        R = G = B = 0;

        int   i = (int)(H * 6);
        T f = H * 6 - i;
        T p = V * (1 - S);
        T q = V * (1 - f * S);
        T t = V * (1 - (1 - f) * S);

        switch (i % 6) {
            case 0: R = V, G = t, B = p; break;
            case 1: R = q, G = V, B = p; break;
            case 2: R = p, G = V, B = t; break;
            case 3: R = p, G = q, B = V; break;
            case 4: R = t, G = p, B = V; break;
            case 5: R = V, G = p, B = q; break;
        }

        dst[oIdx0] = R;
        dst[oIdx1] = G;
        dst[oIdx2] = B;
#else
        T R = src[iIdx0];
        T G = src[iIdx1];
        T B = src[iIdx2];
        T Cmax = fmax(fmax(R, G), B);
        T Cmin = fmin(fmin(R, G), B);
        T delta= Cmax-Cmin;

        T H = 0;

        if (Cmax!=Cmin) {
            if (Cmax==R) H = (G-B)/delta + (G<B ? 6 : 0);
            if (Cmax==G) H = (B-R)/delta + 2;
            if (Cmax==B) H = (R-G)/delta + 4;
            H = H / 6.0f;
        }

        dst[oIdx0] = H;
        dst[oIdx1] = Cmax==0.0f ? 0 : delta/Cmax;
        dst[oIdx2] = Cmax;
#endif
    }
}
