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
#include "transform_interp.hpp"

namespace cpu
{
    template <typename T>
    void calc_transform_inverse(T *txo, const T *txi, const bool perspective)
    {
        if (perspective) {
            txo[0] =   txi[4]*txi[8] - txi[5]*txi[7];
            txo[1] = -(txi[1]*txi[8] - txi[2]*txi[7]);
            txo[2] =   txi[1]*txi[5] - txi[2]*txi[4];

            txo[3] = -(txi[3]*txi[8] - txi[5]*txi[6]);
            txo[4] =   txi[0]*txi[8] - txi[2]*txi[6];
            txo[5] = -(txi[0]*txi[5] - txi[2]*txi[3]);

            txo[6] =   txi[3]*txi[7] - txi[4]*txi[6];
            txo[7] = -(txi[0]*txi[7] - txi[1]*txi[6]);
            txo[8] =   txi[0]*txi[4] - txi[1]*txi[3];

            T det = txi[0]*txo[0] + txi[1]*txo[3] + txi[2]*txo[6];

            txo[0] /= det; txo[1] /= det; txo[2] /= det;
            txo[3] /= det; txo[4] /= det; txo[5] /= det;
            txo[6] /= det; txo[7] /= det; txo[8] /= det;
        }
        else {
            T det = txi[0]*txi[4] - txi[1]*txi[3];

            txo[0] = txi[4] / det;
            txo[1] = txi[3] / det;
            txo[3] = txi[1] / det;
            txo[4] = txi[0] / det;

            txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
            txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
        }
    }

    template <typename T>
    void calc_transform_inverse(T *tmat, const T *tmat_ptr, const bool inverse,
                                const bool perspective, const unsigned transf_len)
    {
        // The way kernel is structured, it expects an inverse
        // transform matrix by default.
        // If it is an forward transform, then we need its inverse
        if(inverse) {
            for(int i = 0; i < (int)transf_len; i++)
                tmat[i] = tmat_ptr[i];
        } else {
            calc_transform_inverse(tmat, tmat_ptr, perspective);
        }
    }

    template<typename T, af_interp_type method>
    void transform_(T *out, const T *in, const float *tf,
                    const af::dim4 &odims, const af::dim4 &idims,
                    const af::dim4 &ostrides, const af::dim4 &istrides,
                    const af::dim4 &tstrides, const bool inverse,
                    const bool perspective)
    {
        dim_t nimages     = idims[2];
        // Multiplied in src/backend/transform.cpp
        dim_t ntransforms = odims[2] / idims[2];

        void (*t_fn)(T *, const T *, const float *, const af::dim4 &,
                     const af::dim4 &, const af::dim4 &,
                     const dim_t, const dim_t, const dim_t, const dim_t, const bool);

        switch(method) {
            case AF_INTERP_NEAREST:
                t_fn = &transform_n;
                break;
            case AF_INTERP_BILINEAR:
                t_fn = &transform_b;
                break;
            case AF_INTERP_LOWER:
                t_fn = &transform_l;
                break;
            default:
                AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                break;
        }

        const int transf_len = (perspective) ? 9 : 6;

        // For each transform channel
        for(int t_idx = 0; t_idx < (int)ntransforms; t_idx++) {
            // Compute inverse if required
            const float *tmat_ptr = tf + t_idx * transf_len;
            float* tmat = new float[transf_len];
            calc_transform_inverse(tmat, tmat_ptr, inverse, perspective, transf_len);

            // Offset for output pointer
            dim_t o_offset = t_idx * nimages * ostrides[2];

            // Do transform for image
            for(int yy = 0; yy < (int)odims[1]; yy++) {
                for(int xx = 0; xx < (int)odims[0]; xx++) {
                    t_fn(out, in, tmat, idims, ostrides, istrides, nimages, o_offset, xx, yy, perspective);
                }
            }
            delete[] tmat;
        }
    }

    template<typename T>
    Array<T> transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const af_interp_type method, const bool inverse, const bool perspective)
    {
        const af::dim4 idims = in.dims();

        Array<T> out = createEmptyArray<T>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                transform_<T, AF_INTERP_NEAREST>
                          (out.get(), in.get(), transform.get(), odims, idims,
                           out.strides(), in.strides(), transform.strides(), inverse,
                           perspective);
                break;
            case AF_INTERP_BILINEAR:
                transform_<T, AF_INTERP_BILINEAR>
                          (out.get(), in.get(), transform.get(), odims, idims,
                           out.strides(), in.strides(), transform.strides(), inverse,
                           perspective);
                break;
            case AF_INTERP_LOWER:
                transform_<T, AF_INTERP_LOWER>
                          (out.get(), in.get(), transform.get(), odims, idims,
                           out.strides(), in.strides(), transform.strides(), inverse,
                           perspective);
                break;
            default:
                AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
                break;
        }

        return out;
    }


#define INSTANTIATE(T)                                                                  \
    template Array<T> transform(const Array<T> &in, const Array<float> &transform,      \
                                const af::dim4 &odims, const af_interp_type method,     \
                                const bool inverse, const bool perspective);


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
    INSTANTIATE(short)
    INSTANTIATE(ushort)
}
