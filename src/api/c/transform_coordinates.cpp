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
#include <common/err_common.hpp>
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
static af_array transform_coordinates(const af_array& tf_, const float d0_, const float d1_)
{
    af::dim4 h_dims(4, 3);
    T h_in[4*3] = { (T)0, (T)0,  (T)d1_, (T)d1_,
                    (T)0, (T)d0_, (T)d0_, (T)0,
                    (T)1, (T)1,  (T)1,  (T)1 };

    const Array<T> tf = getArray<T>(tf_);
    Array<T> in = createHostDataArray<T>(h_dims, h_in);

    std::vector<af_seq> idx(2);
    idx[0] = af_make_seq(0, 2, 1);

    // w = 1.0 / matmul(tf, in(span, 2));
    // iw = matmul(tf, in(span, 2));
    idx[1] = af_make_seq(2, 2, 1);
    Array<T> iw = multiplyIndexed(in, tf, idx);

    // xt = w * matmul(tf, in(span, 0));
    // xt = matmul(tf, in(span, 0)) / iw;
    idx[1] = af_make_seq(0, 0, 1);
    Array<T> xt = arithOp<T, af_div_t>(multiplyIndexed(in, tf, idx), iw, iw.dims());

    // yt = w * matmul(tf, in(span, 1));
    // yt = matmul(tf, in(span, 1)) / iw;
    idx[1] = af_make_seq(1, 1, 1);
    Array<T> yw = arithOp<T, af_div_t>(multiplyIndexed(in, tf, idx), iw, iw.dims());

    // return join(1, xt, yt)
    Array<T> r = join(1, xt, yw);
    return getHandle(r);
}

af_err af_transform_coordinates(af_array *out, const af_array tf, const float d0_, const float d1_)
{
    try {
        const ArrayInfo& tfInfo = getInfo(tf);
        dim4 tfDims = tfInfo.dims();
        ARG_ASSERT(1, (tfDims[0]==3 && tfDims[1]==3 && tfDims.ndims()==2));

        af_array output;
        af_dtype type  = tfInfo.getType();
        switch(type) {
            case f32: output = transform_coordinates<float >(tf, d0_, d1_); break;
            case f64: output = transform_coordinates<double>(tf, d0_, d1_); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
