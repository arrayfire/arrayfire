/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <types.hpp>
#include <af/traits.hpp>

namespace cpu
{
    using std::conditional;
    using std::is_same;

    template<typename T>
    using wtype_t = typename conditional<is_same<T, double>::value, double, float>::type;

    template<typename T>
    using vtype_t = typename conditional<is_complex<T>::value,
                                         T, wtype_t<T>
                                        >::type;

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

        T val = scalar<T>(0.0f);
        // Copy to output
        for(int batch = 0; batch < (int)idims[3]; batch++) {
            dim_t i__ = batch * istrides[3];
            dim_t o__ = batch * ostrides[3];
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
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

        if (xi < -0.0001 || yi < -0.0001 || idims[0] < xi || idims[1] < yi) {
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
                const dim_t o_off = o_offset + i_idx * ostrides[2] + loco;
                out[o_off] = scalar<T>(0.0f);
            }
            return;
        }

        typedef typename dtype_traits<T>::base_type BT;
        typedef wtype_t<BT> WT;
        typedef vtype_t<T> VT;

        const WT grd_x = floor(xi),  grd_y = floor(yi);
        const WT off_x = xi - grd_x, off_y = yi - grd_y;

        dim_t loci = grd_y * istrides[1] + grd_x;

        // Check if pVal and pVal + 1 are both valid indices
        bool condY = (yi < idims[1] - 1);
        bool condX = (xi < idims[0] - 1);

        const T zero = scalar<T>(0.0f);

        // Compute weights used
        const WT wt00 = (1.0 - off_x) * (1.0 - off_y);
        const WT wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
        const WT wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
        const WT wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

        const WT wt = wt00 + wt10 + wt01 + wt11;

        for(int batch = 0; batch < (int)idims[3]; batch++) {
            dim_t i__ = batch * istrides[3];
            dim_t o__ = batch * ostrides[3];
            for(int i_idx = 0; i_idx < (int)nimages; i_idx++) {
                const dim_t i_off = i_idx * istrides[2] + loci + i__;
                const dim_t o_off = o_offset + i_idx * ostrides[2] + loco + o__;
                // Compute Weighted Values
                VT v00 =                    in[i_off] * wt00;
                VT v10 = (condY) ?          in[i_off + istrides[1]] * wt10     : zero;
                VT v01 = (condX) ?          in[i_off + 1] * wt01               : zero;
                VT v11 = (condX && condY) ? in[i_off + istrides[1] + 1] * wt11 : zero;
                VT vo = v00 + v10 + v01 + v11;

                out[o_off] = vo / wt;
            }
        }
    }
}
