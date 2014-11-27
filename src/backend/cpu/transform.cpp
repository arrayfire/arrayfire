/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <transform.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    template <typename T>
    void calc_affine_inverse(T *txo, const T *txi)
    {
        T det = txi[0]*txi[4] - txi[1]*txi[3];

        txo[0] = txi[4] / det;
        txo[1] = txi[3] / det;
        txo[3] = txi[1] / det;
        txo[4] = txi[0] / det;

        txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
        txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
    }

    template <typename T>
    void calc_affine_inverse(T *tmat, const T *tmat_ptr, const bool inverse)
    {
        // The way kernel is structured, it expects an inverse
        // transform matrix by default.
        // If it is an forward transform, then we need its inverse
        if(inverse) {
            for(int i = 0; i < 6; i++)
                tmat[i] = tmat_ptr[i];
        } else {
            calc_affine_inverse(tmat, tmat_ptr);
        }
    }

    template<typename T>
    void transform_n(T *out, const T *in, const float *tmat, const af::dim4 &idims,
                      const af::dim4 &ostrides, const af::dim4 &istrides,
                      const dim_type nimages, const dim_type o_offset,
                      const dim_type xx, const dim_type yy)
    {
        // Compute output index
        const dim_type xi = round(xx * tmat[0]
                                + yy * tmat[1]
                                     + tmat[2]);
        const dim_type yi = round(xx * tmat[3]
                                + yy * tmat[4]
                                     + tmat[5]);

        // Compute memory location of indices
        dim_type loci = (yi * istrides[1] + xi);
        dim_type loco = (yy * ostrides[1] + xx);

        // Copy to output
        for(int i_idx = 0; i_idx < nimages; i_idx++) {
            T val = scalar<T>(0.0f);
            dim_type i_off = i_idx * istrides[2];
            dim_type o_off = o_offset + i_idx * ostrides[2];

            if (xi < idims[0] && yi < idims[1] && xi >= 0 && yi >= 0)
                val = in[i_off + loci];

            out[o_off + loco] = val;
        }
    }

    template<typename T>
    void transform_b(T *out, const T *in, const float *tmat, const af::dim4 &idims,
                      const af::dim4 &ostrides, const af::dim4 &istrides,
                      const dim_type nimages, const dim_type o_offset,
                      const dim_type xx, const dim_type yy)
    {
        dim_type loco = (yy * ostrides[1] + xx);
        // Compute input index
        const float xi = xx * tmat[0]
                       + yy * tmat[1]
                            + tmat[2];
        const float yi = xx * tmat[3]
                       + yy * tmat[4]
                            + tmat[5];

        if (xi < 0 || yi < 0 || idims[0] < xi || idims[1] < yi) {
            for(int i_idx = 0; i_idx < nimages; i_idx++) {
                const dim_type o_off = o_offset + i_idx * ostrides[2] + loco;
                out[o_off] = scalar<T>(0.0f);
            }
            return;
        }

        const float grd_x = floor(xi),  grd_y = floor(yi);
        const float off_x = xi - grd_x, off_y = yi - grd_y;

        dim_type loci = grd_y * istrides[1] + grd_x;

        // Check if pVal and pVal + 1 are both valid indices
        bool condY = (yi < idims[1] - 1);
        bool condX = (xi < idims[0] - 1);

        // Compute weights used
        float wt00 = (1.0 - off_x) * (1.0 - off_y);
        float wt10 = (condY) ? (1.0 - off_x) * (off_y)     : 0;
        float wt01 = (condX) ? (off_x) * (1.0 - off_y)     : 0;
        float wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

        float wt = wt00 + wt10 + wt01 + wt11;

        for(int i_idx = 0; i_idx < nimages; i_idx++) {
            const dim_type i_off = i_idx * istrides[2] + loci;
            const dim_type o_off = o_offset + i_idx * ostrides[2] + loco;
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

    template<typename T, af_interp_type method>
    void transform_(T *out, const T *in, const float *tf,
                    const af::dim4 &odims, const af::dim4 &idims,
                    const af::dim4 &ostrides, const af::dim4 &istrides,
                    const af::dim4 &tstrides, const bool inverse)
    {
        dim_type nimages     = idims[2];
        // Multiplied in src/backend/transform.cpp
        dim_type ntransforms = odims[2] / idims[2];

        void (*t_fn)(T *, const T *, const float *, const af::dim4 &,
                     const af::dim4 &, const af::dim4 &,
                     const dim_type, const dim_type, const dim_type, const dim_type);

        switch(method) {
            case AF_INTERP_NEAREST:
                t_fn = &transform_n;
                break;
            case AF_INTERP_BILINEAR:
                t_fn = &transform_b;
                break;
            default:
                AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                break;
        }


        // For each transform channel
        for(int t_idx = 0; t_idx < ntransforms; t_idx++) {
            // Compute inverse if required
            const float *tmat_ptr = tf + t_idx * 6;
            float tmat[6];
            calc_affine_inverse(tmat, tmat_ptr, inverse);

            // Offset for output pointer
            dim_type o_offset = t_idx * nimages * ostrides[2];

            // Do transform for image
            for(int yy = 0; yy < odims[1]; yy++) {
                for(int xx = 0; xx < odims[0]; xx++) {
                    t_fn(out, in, tmat, idims, ostrides, istrides, nimages, o_offset, xx, yy);
                }
            }
        }
    }

    template<typename T>
    Array<T>* transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const af_interp_type method, const bool inverse)
    {
        const af::dim4 idims = in.dims();

        Array<T> *out = createEmptyArray<T>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                transform_<T, AF_INTERP_NEAREST>
                          (out->get(), in.get(), transform.get(), odims, idims,
                           out->strides(), in.strides(), transform.strides(), inverse);
                break;
            case AF_INTERP_BILINEAR:
                transform_<T, AF_INTERP_BILINEAR>
                          (out->get(), in.get(), transform.get(), odims, idims,
                           out->strides(), in.strides(), transform.strides(), inverse);
                break;
            default:
                AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                break;
        }

        return out;
    }


#define INSTANTIATE(T)                                                                          \
    template Array<T>* transform(const Array<T> &in, const Array<float> &transform,             \
                                 const af::dim4 &odims, const af_interp_type method,            \
                                 const bool inverse);                                           \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}
