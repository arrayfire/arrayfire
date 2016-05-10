/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <err_cpu.hpp>

namespace cpu
{
namespace kernel
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
void transform(Array<T> output, const Array<T> input,
               const Array<float> transform, const bool inverse,
               const bool perspective)
{
    const af::dim4 idims    = input.dims();
    const af::dim4 odims    = output.dims();
    const af::dim4 tdims    = transform.dims();
    const af::dim4 istrides = input.strides();
    const af::dim4 ostrides = output.strides();

    T * out = output.get();
    const T * in = input.get();
    const float* tf = transform.get();

    int nImg2 = idims[2];
    int nImg3 = idims[3];
    int nTfs2 = tdims[2];
    int nTfs3 = tdims[3];

    void (*t_fn)(T *, const T *, const float *, const af::dim4 &,
                 const af::dim4 &, const af::dim4 &,
                 const dim_t, const dim_t, const dim_t, const dim_t,
                 const bool);

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

    int batchImg2 = 1;
    int batchImg3 = 1;
    if(nImg2 != nTfs2 && nImg2 > 1)
        batchImg2 = nImg2;
    if(nImg3 != nTfs3 && nImg3 > 1)
        batchImg3 = nImg3;

    af::dim4 idims_ = idims;
    idims_[3] = batchImg3;
    idims_[2] = batchImg2;
    const dim_t nimages  = batchImg2;

    // For each transform channel
    for(int t_idx3 = 0; t_idx3 < nTfs3; t_idx3++) {
        int offset3 = 0;
        int i_offset3 = 0;
        if(nTfs3 > 1) {       // Not Image Batched
            offset3 = t_idx3 * ostrides[3];
            if(nImg3 > 1) i_offset3 = t_idx3 * istrides[3]; // One to one batching
        }

        for(int t_idx2 = 0; t_idx2 < nTfs2; t_idx2++) {

            // Compute inverse if required
            const float *tmat_ptr = tf + (t_idx3 * nTfs2 + t_idx2) * transf_len;
            float* tmat = new float[transf_len];
            calc_transform_inverse(tmat, tmat_ptr, inverse, perspective, transf_len);

            int offset2 = 0;
            int i_offset2 = 0;
            if(nTfs2 > 1) {       // Not Image Batched
                offset2 = t_idx2 * ostrides[2];
                if(nImg2 > 1) i_offset2 = t_idx2 * istrides[2]; // One to one batching
            }

            int i_offset = i_offset3 + i_offset2;

            // Do transform for image
            for(int yy = 0; yy < (int)odims[1]; yy++) {
                for(int xx = 0; xx < (int)odims[0]; xx++) {
                    t_fn(out, in + i_offset, tmat, idims_, ostrides, istrides,
                         nimages, offset3 + offset2, xx, yy, perspective);
                }
            }
            delete[] tmat;
        }
    }
}

}
}
