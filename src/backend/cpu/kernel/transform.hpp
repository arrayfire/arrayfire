/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_cpu.hpp>
#include <af/traits.hpp>
#include <type_traits>
#include "interp.hpp"

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void calc_transform_inverse(T *txo, const T *txi, const bool perspective) {
    if (perspective) {
        txo[0] = txi[4] * txi[8] - txi[5] * txi[7];
        txo[1] = -(txi[1] * txi[8] - txi[2] * txi[7]);
        txo[2] = txi[1] * txi[5] - txi[2] * txi[4];

        txo[3] = -(txi[3] * txi[8] - txi[5] * txi[6]);
        txo[4] = txi[0] * txi[8] - txi[2] * txi[6];
        txo[5] = -(txi[0] * txi[5] - txi[2] * txi[3]);

        txo[6] = txi[3] * txi[7] - txi[4] * txi[6];
        txo[7] = -(txi[0] * txi[7] - txi[1] * txi[6]);
        txo[8] = txi[0] * txi[4] - txi[1] * txi[3];

        T det = txi[0] * txo[0] + txi[1] * txo[3] + txi[2] * txo[6];

        txo[0] /= det;
        txo[1] /= det;
        txo[2] /= det;
        txo[3] /= det;
        txo[4] /= det;
        txo[5] /= det;
        txo[6] /= det;
        txo[7] /= det;
        txo[8] /= det;
    } else {
        T det = txi[0] * txi[4] - txi[1] * txi[3];

        txo[0] = txi[4] / det;
        txo[1] = txi[3] / det;
        txo[3] = txi[1] / det;
        txo[4] = txi[0] / det;

        txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
        txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
    }
}

template<typename T>
void calc_transform_inverse(T *tmat, const T *tmat_ptr, const bool inverse,
                            const bool perspective, const unsigned transf_len) {
    // The way kernel is structured, it expects an inverse
    // transform matrix by default.
    // If it is an forward transform, then we need its inverse
    if (inverse) {
        for (int i = 0; i < (int)transf_len; i++) tmat[i] = tmat_ptr[i];
    } else {
        calc_transform_inverse(tmat, tmat_ptr, perspective);
    }
}

template<typename T, int order>
void transform(Param<T> output, CParam<T> input, CParam<float> transform,
               const bool inverse, const bool perspective,
               af_interp_type method) {
    typedef typename af::dtype_traits<T>::base_type BT;
    typedef wtype_t<BT> WT;

    const af::dim4 idims    = input.dims();
    const af::dim4 odims    = output.dims();
    const af::dim4 tdims    = transform.dims();
    const af::dim4 tstrides = transform.strides();
    const af::dim4 istrides = input.strides();
    const af::dim4 ostrides = output.strides();

    T *out          = output.get();
    const float *tf = transform.get();

    int batch_size = 1;
    if (idims[2] != tdims[2]) batch_size = idims[2];

    Interp2<T, WT, order> interp;
    for (int idw = 0; idw < (int)odims[3]; idw++) {
        dim_t out_offw = idw * ostrides[3];
        dim_t in_offw  = (idims[3] > 1) * idw * istrides[3];
        dim_t tf_offw  = (tdims[3] > 1) * idw * tstrides[3];

        for (int idz = 0; idz < (int)odims[2]; idz += batch_size) {
            dim_t out_offzw = out_offw + idz * ostrides[2];
            dim_t in_offzw  = in_offw + (idims[2] > 1) * idz * istrides[2];
            dim_t tf_offzw  = tf_offw + (tdims[2] > 1) * idz * tstrides[2];

            const float *tptr = tf + tf_offzw;

            float tmat[9];
            calc_transform_inverse(tmat, tptr, inverse, perspective,
                                   perspective ? 9 : 6);

            for (int idy = 0; idy < (int)odims[1]; idy++) {
                for (int idx = 0; idx < (int)odims[0]; idx++) {
                    WT xidi = idx * tmat[0] + idy * tmat[1] + tmat[2];
                    WT yidi = idx * tmat[3] + idy * tmat[4] + tmat[5];

                    if (perspective) {
                        WT W = idx * tmat[6] + idy * tmat[7] + tmat[8];
                        xidi /= W;
                        yidi /= W;
                    }

                    // FIXME: Nearest and lower do not do clamping, but other
                    // methods do Make it consistent
                    bool clamp = order != 1;
                    bool condX = xidi >= -0.0001 && xidi < idims[0];
                    bool condY = yidi >= -0.0001 && yidi < idims[1];

                    int ooff = out_offzw + idy * ostrides[1] + idx;
                    if (condX && condY) {
                        interp(output, ooff, input, in_offzw, xidi, yidi,
                               method, batch_size, clamp);
                    } else {
                        for (int n = 0; n < batch_size; n++) {
                            out[ooff + n * ostrides[2]] = scalar<T>(0);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
