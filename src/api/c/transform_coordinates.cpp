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
#include <af/image.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <convolve.hpp>
#include <arith.hpp>
#include <blas.hpp>
#include <join.hpp>
#include <vector>

using af::dim4;
using namespace detail;

template<typename T>
Array<T> multiplyIndexed(const Array<T> &lhs, const Array<T> &rhs, std::vector<af_seq> idx)
{
    return matmul(lhs, createSubArray(rhs, idx), AF_MAT_NONE, AF_MAT_NONE);
}

template<typename T>
static af_array transform_coordinates(const af_array& tf, const float d0, const float d1)
{
    af::dim4 h_dims(4, 3);
    T h_in[4*3] = { (T)0, (T)0,  (T)d1, (T)d1,
                    (T)0, (T)d0, (T)d0, (T)0,
                    (T)1, (T)1,  (T)1,  (T)1 };

    const Array<T> TF = getArray<T>(tf);
    Array<T> IN = createHostDataArray<T>(h_dims, h_in);

    std::vector<af_seq> idx(2);
    idx[0] = af_make_seq(0, 2, 1);

    // w = 1.0 / matmul(TF, IN(span, 2));
    // iw = matmul(TF, IN(span, 2));
    idx[1] = af_make_seq(2, 2, 1);
    Array<T> IW = multiplyIndexed(IN, TF, idx);

    // xt = w * matmul(TF, IN(span, 0));
    // xt = matmul(TF, IN(span, 0)) / iw;
    idx[1] = af_make_seq(0, 0, 1);
    Array<T> XT = arithOp<T, af_div_t>(multiplyIndexed(IN, TF, idx), IW, IW.dims());

    // yt = w * matmul(TF, IN(span, 1));
    // yt = matmul(TF, IN(span, 1)) / iw;
    idx[1] = af_make_seq(1, 1, 1);
    Array<T> YT = arithOp<T, af_div_t>(multiplyIndexed(IN, TF, idx), IW, IW.dims());

    // return join(1, xt, yt)
    Array<T> R = join(1, XT, YT);
    return getHandle(R);
}

af_err af_transform_coordinates(af_array *out, const af_array tf, const float d0, const float d1)
{
    try {
        const ArrayInfo& tfInfo = getInfo(tf);
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
