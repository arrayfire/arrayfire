/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/vision.h>
#include <af/image.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/data.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <convolve.hpp>
#include <arith.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_array transform_coordinates(const af_array& tf, const float d0, const float d1)
{
    dim_t in_dims[2] = { 4, 3 };
    T h_in[4*3] = { (T)0, (T)0,  (T)d1, (T)d1,
                    (T)0, (T)d0, (T)d0, (T)0,
                    (T)1, (T)1,  (T)1,  (T)1 };

    af_array in  = 0;
    af_array w   = 0;
    af_array tmp = 0;
    af_array xt  = 0;
    af_array yt  = 0;
    af_array t   = 0;

    AF_CHECK(af_create_array(&in, h_in, 2, in_dims, (af_dtype) af::dtype_traits<T>::af_type));

    af_array tfIdx = 0;
    af_index_t tfIndexs[2];
    tfIndexs[0].isSeq = true;
    tfIndexs[1].isSeq = true;
    tfIndexs[0].idx.seq = af_make_seq(0, 2, 1);
    tfIndexs[1].idx.seq = af_make_seq(2, 2, 1);
    AF_CHECK(af_index_gen(&tfIdx, tf, 2, tfIndexs));

    AF_CHECK(af_matmul(&tmp, in, tfIdx, AF_MAT_NONE, AF_MAT_NONE));
    T h_w[4] = { 1, 1, 1, 1 };
    dim_t w_dims = 4;
    AF_CHECK(af_create_array(&w, h_w, 1, &w_dims, (af_dtype) af::dtype_traits<T>::af_type));
    AF_CHECK(af_div(&w, w, tmp, false));

    tfIndexs[1].idx.seq = af_make_seq(0, 0, 1);
    AF_CHECK(af_index_gen(&tfIdx, tf, 2, tfIndexs));
    AF_CHECK(af_matmul(&tmp, in, tfIdx, AF_MAT_NONE, AF_MAT_NONE));
    AF_CHECK(af_mul(&xt, tmp, w, false));

    tfIndexs[1].idx.seq = af_make_seq(1, 1, 1);
    AF_CHECK(af_index_gen(&tfIdx, tf, 2, tfIndexs));
    AF_CHECK(af_matmul(&tmp, in, tfIdx, AF_MAT_NONE, AF_MAT_NONE));
    AF_CHECK(af_mul(&yt, tmp, w, false));

    AF_CHECK(af_join(&t, 1, xt, yt));

    AF_CHECK(af_release_array(w));
    AF_CHECK(af_release_array(tmp));
    AF_CHECK(af_release_array(xt));
    AF_CHECK(af_release_array(yt));

    return t;
}

af_err af_transform_coordinates(af_array *out, const af_array tf, const float d0, const float d1)
{
    try {
        ArrayInfo tfInfo = getInfo(tf);
        dim4 tfDims = tfInfo.dims();
        ARG_ASSERT(1, (tfDims[0]==3 && tfDims[1]==3 && tfDims.ndims()==2));

        af_array output;
        af_dtype type  = tfInfo.getType();
        switch(type) {
            case f32: output = transform_coordinates<float >(tf, d0, d1); break;
            case f64: output = transform_coordinates<double>(tf, d0, d1); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
