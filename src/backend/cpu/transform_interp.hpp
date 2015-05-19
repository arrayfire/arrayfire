/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace cpu
{
    template<typename T>
    void transform_n(T *out, const T *in, const float *tmat, const af::dim4 &idims,
                      const af::dim4 &ostrides, const af::dim4 &istrides,
                      const dim_t nimages, const dim_t o_offset,
                      const dim_t xx, const dim_t yy)
    {
        // Compute output index
        const dim_t xi = round(xx * tmat[0]
                                + yy * tmat[1]
                                     + tmat[2]);
        const dim_t yi = round(xx * tmat[3]
                                + yy * tmat[4]
                                     + tmat[5]);

        // Compute memory location of indices
        dim_t loci = (yi * istrides[1] + xi);
        dim_t loco = (yy * ostrides[1] + xx);

        // Copy to output
        for(int batch = 0; batch < (int)idims[3]; batch++) {
            dim_t i__ = batch * istrides[3];
            dim_t o__ = batch * ostrides[3];
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
                T val = scalar<T>(0.0f);
                dim_t i_off = i_idx * istrides[2] + i__;
                dim_t o_off = o_offset + i_idx * ostrides[2] + o__;

                if (xi < idims[0] && yi < idims[1] && xi >= 0 && yi >= 0)
                    val = in[i_off + loci];

                out[o_off + loco] = val;
            }
        }
    }

    template<typename T>
    void transform_b(T *out, const T *in, const float *tmat, const af::dim4 &idims,
                      const af::dim4 &ostrides, const af::dim4 &istrides,
                      const dim_t nimages, const dim_t o_offset,
                      const dim_t xx, const dim_t yy)
    {
        dim_t loco = (yy * ostrides[1] + xx);
        // Compute input index
        const float xi = xx * tmat[0]
                       + yy * tmat[1]
                            + tmat[2];
        const float yi = xx * tmat[3]
                       + yy * tmat[4]
                            + tmat[5];

        if (xi < 0 || yi < 0 || idims[0] < xi || idims[1] < yi) {
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
                const dim_t o_off = o_offset + i_idx * ostrides[2] + loco;
                out[o_off] = scalar<T>(0.0f);
            }
            return;
        }

        const float grd_x = floor(xi),  grd_y = floor(yi);
        const float off_x = xi - grd_x, off_y = yi - grd_y;

        dim_t loci = grd_y * istrides[1] + grd_x;

        // Check if pVal and pVal + 1 are both valid indices
        bool condY = (yi < idims[1] - 1);
        bool condX = (xi < idims[0] - 1);

        // Compute weights used
        float wt00 = (1.0 - off_x) * (1.0 - off_y);
        float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
        float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
        float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

        float wt = wt00 + wt10 + wt01 + wt11;

        for(int batch = 0; batch < (int)idims[3]; batch++) {
            dim_t i__ = batch * istrides[3];
            dim_t o__ = batch * ostrides[3];
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
                const dim_t i_off = i_idx * istrides[2] + loci + i__;
                const dim_t o_off = o_offset + i_idx * ostrides[2] + loco + o__;
                // Compute Weighted Values
                T zero = scalar<T>(0.0f);
                T v00 =                    wt00 * in[i_off];
                T v10 = (condY) ?          wt10 * in[i_off + istrides[1]]     : zero;
                T v01 = (condX) ?          wt01 * in[i_off + 1]               : zero;
                T v11 = (condX && condY) ? wt11 * in[i_off + istrides[1] + 1] : zero;
                T vo = v00 + v10 + v01 + v11;

                out[o_off] = (vo / wt);
            }
        }
    }
}
